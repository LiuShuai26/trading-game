import sys

sys.path.append("/home/shuai/trading-game/")


def ppo(env_fn):
    env = env_fn()

    def test():
        print("===================start test======================")
        o, r, d, test_ret, test_len = env.reset(start_day=62, start_skip=0, duration=None), 0, False, 0, 0
        # o, r, d, test_ret, test_len = env.reset(start_day=62, start_skip=0, duration=None, burn_in=0), 0, False, 0, 0
        test_target_bias, test_reward_target_bias, test_reward_score, test_reward_apnum = 0, 0, 0, 0
        while True:
            a = env.baseline_policy(o)
            o, r, d, info = env.step(a)
            test_ret += r
            test_len += 1
            test_target_bias += info["target_bias"]
            test_reward_target_bias += info["reward_target_bias"]
            test_reward_score += info["reward_score"]
            test_reward_apnum += info["reward_ap_num"]

            if d or test_len == 3000:  # for fast debug
                # if d:
                print("Day", 62, "len:", test_len)
                break
        print("===================end test======================")
        # after test we need reset the env
        o, ep_ret, ep_len = env.reset(), 0, 0
        # o, ep_ret, ep_len = env.reset(), 0, 0

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_target_bias, ep_reward_target_bias, ep_reward_score, ep_apnum = 0, 0, 0, 0
    ep_score = 0

    print(o.shape)

    while True:
        # print("------------", ep_len)
        # action = env.baseline_policy(o)
        action = 0
        o, r, d, info = env.step(action)
        ep_ret += r
        ep_len += 1
        ep_target_bias += info["target_bias"]
        ep_reward_target_bias += info["reward_target_bias"]
        ep_reward_score += info["reward_score"]
        ep_apnum += info["reward_ap_num"]
        ep_score += info["one_step_score"]

        # print("info score:", info["score"], "ep score:", ep_score)

        terminal = d or (ep_len == 3000)
        if terminal:
            print("len:", ep_len, "info score:", info["score"], "ep score:", ep_score)
            print("done")
            o, ep_ret, ep_len = env.reset(), 0, 0
            ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_apnum = 0, 0, 0, 0, 0
            ep_score = 0

            test()


action_scheme_id = 15

num_stack = 3

delay_len = 30
target_scale = 1
score_scale = 1.5
ap = 0.5
target_clip = 0
burn_in = 3000

max_ep_len = 3000

start_day = None
start_skip = None
duration = None

from trading_env import TradingEnv, FrameStack

env = TradingEnv(action_scheme_id=action_scheme_id, select_obs=True, render=False,
                 max_ep_len=max_ep_len, delay_len=delay_len,
                 target_scale=target_scale, score_scale=score_scale, action_punish=ap, target_clip=target_clip,
                 start_day=start_day, start_skip=start_skip, duration=duration, burn_in=burn_in)

if num_stack > 1:
    env = FrameStack(env, num_stack)

ppo(lambda: env)
