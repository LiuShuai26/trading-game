import pandas as pd
import env
import matplotlib.pyplot as plt
import numpy as np


env = env.TradingEnv()


targets = []
acts = []

for i in range(20):
    obs = env.reset()
    step = 1
    score = []
    epret = 0
    while True:
        # action = env.action_space.sample()

        if obs[24] > obs[25]:
            action = 5
        elif obs[24] < obs[25]:
            action = 10
        else:
            action = 0
        # action = 0
        obs, reward, done, info = env.step(action)
        step += 1
        epret += reward
        # if step > 400:
        #     score.append(info['score'])
        # score.append(env.rewards[3])

        # if done:
        if done or step == 1000:
            # plt.plot(score)
            # plt.show()
            print(epret)
            # all_data = env.all_data
            # all_data_df = pd.DataFrame(all_data)
            # print(all_data_df.tail())
            # all_data_df.to_csv("/home/shuai/day1-test-action0312.csv", index=False)
            # print("data save!")
            break
