import pandas as pd
# from env import TradingEnv
import matplotlib.pyplot as plt
import numpy as np
import gym
# from gym.wrappers.frame_stack import FrameStack
from trading_env import TradingEnv, FrameStack, EnvWrapper

# env = env.TradingEnv(dataset_size=1, num_stack=3)
env = TradingEnv(action_scheme_id=15)
env = FrameStack(env, 3)
env = EnvWrapper(env, delay_len=30, target_scale=1, score_scale=1, ap=0.5, target_clip=3, env_skip=True)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape
print(obs_dim)
print(act_dim)
print(env.observation_space, env.action_space)


# env = gym.make('CartPole-v0')

#
# obs_dim = env.observation_space.shape
# act_dim = env.action_space.shape
# print(obs_dim)
# print(act_dim)
# print(env.observation_space, env.action_space)

for i in range(1):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_apnum = 0, 0, 0, 0, 0
    # print(o)
    step = 1
    while True:
        action = env.action_space.sample()
        # action = 0
        obs, reward, done, info = env.step(action)
        ep_ret += r
        ep_len += 1
        ep_target_bias += info["target_bias"]
        ep_reward_target_bias += info["reward_target_bias"]
        ep_score += info["one_step_score"]
        ep_reward_score += info["reward_score"]
        ep_apnum += info["ap_num"]
        step += 1

        # if done:
        if done or step == 4000:
            print("DONE")
            print("ep_target_bias", ep_target_bias)
            print("ep_reward_target_bias", ep_reward_target_bias)
            print("ep_score", ep_score)
            print("ep_reward_score", ep_reward_score)
            print("ep_apnum", ep_apnum)
            break
