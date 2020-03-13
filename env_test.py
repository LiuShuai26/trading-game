import pandas as pd
import env
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.wrappers.frame_stack import FrameStack


env = env.TradingEnv(num_stack=3)
# env = gym.make('CartPole-v0')
# env = FrameStack(env, 3)

print(env.observation_space)

targets = []
acts = []

for i in range(1):
    obs = env.reset()
    print(obs)
    print(np.array(obs).reshape(1, -1))

    step = 1
    while True:
        action = env.action_space.sample()
        # action = 0
        obs, reward, done, info = env.step(action)
        print(np.array(obs))
        step += 1

        # if done:
        if done or step == 10:
            print("DONE")
            break
