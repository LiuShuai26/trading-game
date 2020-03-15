import pandas as pd
import env
import matplotlib.pyplot as plt
import numpy as np


env = env.TradingEnv()


targets = []
acts = []

for i in range(10):
    obs = env.reset()
    step = 1
    score = []
    epret = 0
    epscore = 0
    last_score = 0
    while True:
        # action = env.action_space.sample()

        if obs[24] > obs[25]:
            action = 6
        elif obs[24] < obs[25]:
            action = 9
        else:
            action = 0

        if step < 1000:
            action = 0
        # action = 0
        obs, reward, done, info = env.step(action)
        step += 1
        epret += reward
        epscore += (info['score']-last_score)/100
        # if step > 400:
        #     score.append(info['score'])
        if step > 1000:
            score.append((info['score']-last_score)/100)
        last_score = info['score']

        # if done:
        if done or step == 4000:
            # plt.plot(score)
            # plt.show()
            print(epscore)
            # print(epret)
            # all_data = env.all_data
            # all_data_df = pd.DataFrame(all_data)
            # print(all_data_df.tail())
            # all_data_df.to_csv("/home/shuai/day1-test-action0312.csv", index=False)
            # print("data save!")
            break
