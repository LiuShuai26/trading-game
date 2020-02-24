import env

env = env.TradingEnv()

while True:
    for i in range(1, 63):
        obs = env.reset(start_day=33)
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
            print(obs)
            step += 1
            # print('obs=', obs, 'reward=', reward, 'done=', done)
            # print('reward=', reward, 'profit=', info['profit'])

            if done:
                print("TradingDay", info["TradingDay"], "step:", step)
                break
