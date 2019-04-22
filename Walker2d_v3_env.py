import gym
env = gym.make('Walker2d-v3')
print(env.action_space) 
# Box(6,)

print(env.action_space.high)
print(env.action_space.low)

# [1. 1. 1. 1. 1. 1.]
# [-1. -1. -1. -1. -1. -1.]

print(env.observation_space) 
# Box(17,)

print(env.observation_space.high)
# inf

print(env.observation_space.low)
# -inf

input("Press Enter to continue...")

for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
