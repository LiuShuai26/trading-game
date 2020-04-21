from trading_env import TradingEnv
from wrapper import EnvWrapper


start_day = 32
start_skip = 0
duration = 30000
burn_in = 1000

env = TradingEnv(action_scheme_id=15)
env = EnvWrapper(env, delay_len=30, target_scale=1, score_scale=1, action_punish=0.5, target_clip=0,
                 start_day=start_day, start_skip=start_skip, duration=duration, burn_in=burn_in, target_delay=True)


max_len = 3000

test_times = 1000

ep_score = []
for i in range(test_times):
    obs = env.reset()
    step = 1
    while True:
        action = env.baseline_policy(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        if done or step == max_len:
            ep_score.append(info['score'])
            print(i, "score:", info['score'], "ave score:", sum(ep_score)/(i+1))
            break
print("test len:", max_len, "ave score:", sum(ep_score)/test_times)
