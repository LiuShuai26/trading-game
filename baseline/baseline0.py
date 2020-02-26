import pandas as pd
import env

env = env.TradingEnv()


targets = []
acts = []

for i in range(1):
    obs = env.reset(start_day=1, skip_step=10000, analyse=True)
    step = 1
    while True:
        # action = env.action_space.sample()

        if obs[24] > obs[25]:
            action = 7
        elif obs[24] < obs[25]:
            action = 8
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        step += 1

        # if done:
        if done or step == 1000:
            all_data = env.all_data
            all_data_df = pd.DataFrame(all_data)
            print(all_data_df.tail())
            all_data_df.to_csv("/home/shuai/e-day1-780a-1000s.csv", index=False)
            print("data save!")
            break
