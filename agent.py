from __future__ import division
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Refactored from https://github.com/andrewliao11/pytorch-a3c-mujoco

class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0

    def action_train(self):
        if self.args.model == 'CONV':
            self.state = self.state.unsqueeze(0)

        value, mu, sigma, (self.hx, self.cx) = self.model(
            (Variable(self.state), (self.hx, self.cx)))
        mu = torch.clamp(mu, -1.0, 1.0)
        sigma = F.softplus(sigma) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        eps = Variable(eps)
        pi = Variable(pi)

        action = (mu + sigma.sqrt() * eps).data
        act = Variable(action)
        prob = normal(act, mu, sigma)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(
            action.cpu().numpy()[0])
        reward = max(min(float(reward), 1.0), -1.0)
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        self.values.append(value)
        self.rewards.append(reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                self.cx = Variable(torch.zeros(1, 128))
                self.hx = Variable(torch.zeros(1, 128))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
           
            if self.args.model == 'CONV':
                self.state = self.state.unsqueeze(0)
            
            value, mu, sigma, (self.hx, self.cx) = self.model(
                (Variable(self.state), (self.hx, self.cx)))
        mu = torch.clamp(mu.data, -1.0, 1.0)
        action = mu.cpu().numpy()[0]
        state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self

## Utilities

def normal(x, mu, sigma):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b