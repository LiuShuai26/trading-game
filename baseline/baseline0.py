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
    while True:
        # action = env.action_space.sample()

        # if obs[24] > obs[25]:
        #     action = 7
        # elif obs[24] < obs[25]:
        #     action = 8
        # else:
        #     action = 0
        action = 0
        obs, reward, done, info = env.step(action)
        step += 1
        score.append(info['score'])

        # if done:
        if done or step == 1000:
            # all_data = env.all_data
            # all_data_df = pd.DataFrame(all_data)
            # print(all_data_df.describe()['score'].iloc[7])
            score = np.array(score)
            plt.plot(score)
            plt.show()
            # print(all_data_df.tail())
            # all_data_df.to_csv("/home/shuai/day1-078-action.csv", index=False)
            # all_data_df.to_csv("/home/shuai/day1-078-action.csv", index=True)
            # print("data save!")
            break
