import pandas as pd
import env

env = env.TradingEnv()


for i in range(1):
    obs = env.reset(start_day=5, analyse=True)
    step = 1
    while True:
        action = env.action_space.sample()
        # if step < 100000:
        #     action = 3
        # else:
        #     action = 0
        # print("Step {}".format(step))
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        step += 1
        # print('obs=', obs, 'reward=', reward, 'done=', done)
        # print('reward=', reward, 'profit=', info['profit'])

        if done or step == 1000:
            all_data = env.all_data
            all_data_df = pd.DataFrame(all_data)
            print(all_data_df.tail())
            all_data_df.to_csv("/home/shuai/e-day5-random-action-s1000.csv", index=False)
            print("data save!")
            break
