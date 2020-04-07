import pandas as pd
import env
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.wrappers.frame_stack import FrameStack


env = env.TradingEnv(dataset_size=1, num_stack=3)
# env = gym.make('CartPole-v0')
# env = FrameStack(env, 3)

# obs_dim = env.observation_space.shape
# act_dim = env.action_space.shape
# print(obs_dim)
# print(act_dim)
# print(env.observation_space, env.action_space)

for i in range(10):
    obs = env.reset()
    # print(obs)
    step = 1
    while True:
        action = env.action_space.sample()
        # action = 0
        obs, reward, done, info = env.step(action)
        print(info['one_step_score'])
        print(info['reward_target_bias'])
        print("---")
        # print(info)
        # print(env.ap)
        step += 1

        # if done:
        if done or step == 4000:
            print("DONE")
            break
