#TODO : Appropriation du code : ok
#TODO : Reseau du papier google : 2d, double actor output ?
#TODO : Trouver les paramètre pour Walker : ok ?
#TODO : PPO
#TODO : RMSprop shared ok
#TODO : 
#TODO : Walker2d-v3, conv*2 : RMSprop et Adam, 8, 16, 32 workers
#TODO : 

from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import A3C_CONV
from train import train
from test import test
from shared_statistics import SharedRMSprop, SharedAdam
import time

# Inspired by the Universe Starter Agent https://github.com/openai/universe-starter-agent

parser = argparse.ArgumentParser(description='A3C_MODEL_TRAINING')
parser.add_argument(
    '--env',
    default='Walker2d-v3',
    metavar='ENV',
    help='environment to train on')
parser.add_argument(
    '--model',
    default='CONV',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of RMSprop or Adam')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer with our without shared statistics.')
parser.add_argument(
    '--workers',
    type=int,
    default=32,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=10,
    metavar='NS',
    help='number of n-steps (aka. t_max) (default: 10)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run if the high score is matched or bested')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models to')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=1,
    metavar='SF',
    help='Choose number of observations to stack')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')


# Based on Hogwild! https://github.com/pytorch/examples/tree/master/mnist_hogwild

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = create_env(args.env, args)

    # Model selection
    if args.model == 'CONV':
        shared_model = A3C_CONV(env.observation_space.shape[0], env.action_space)
    
    # Load trained model if needed
    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    # Select optimizer
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    # Create parallel actor-learner
    processes = []
    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    
    for p in processes:
        time.sleep(0.1)
        p.join()
