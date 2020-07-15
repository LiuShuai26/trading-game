import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


def ppo(env_fn, data_v):
    env = env_fn()

    def test():
        print("Start test...")
        get_action = env.policy_069
        if data_v == "r19":
            start_test = 91
        else:
            start_test = 51
        for start_day in range(start_test, 8 + start_test):
            o, r, d, test_ret, test_len = env.reset(start_day=start_day), 0, False, 0, 0
            test_target_bias, test_reward_target_bias, test_reward_score, test_reward_ap = 0, 0, 0, 0
            while True:
                a = get_action()
                o, r, d, info = env.step(a)
                test_ret += r
                test_len += 1
                test_target_bias += info["target_bias"]
                test_reward_target_bias += info["reward_target_bias"]
                test_reward_score += info["reward_score"]
                test_reward_ap += info["reward_ap"]

                if d or test_len == 3000:  # for fast debug
                    # if d:
                    print("Day", start_day, "Len:", test_len, "Profit:", info["profit"], "Score:", info["score"])
                    break
        # after test we need reset the env
        o, ep_ret, ep_len = env.reset(), 0, 0

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_reward_profit, ep_reward_ap = 0, 0, 0, 0, 0, 0

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
        ep_reward_ap += info["reward_ap"]
        ep_score += info["one_step_score"]

        # print("info score:", info["score"], "ep score:", ep_score)

        terminal = d or (ep_len == 3000)
        if terminal:
            print("len:", ep_len, "info score:", info["score"], "ep score:", ep_score)
            print("episode done.")
            o = env.reset()
            ep_ret, ep_len = 0, 0
            ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_reward_ap = 0, 0, 0, 0, 0

            test()


data_v = "r19"
model = 'mlp'

action_scheme_id = 15
obs_dim = 26

num_stack = 1

delay_len = 30
target_scale = 1
score_scale = 1.5
profit_scale = 0
ap = 0.5
target_clip = 0
burn_in = 3000
auto_follow = 0
max_ep_len = 3000

if data_v == "r19":
    trainning_set = 90
else:
    trainning_set = 50

from game_env.trading_env import TradingEnv, FrameStack
from game_env.wrapper import EnvWrapper

env = TradingEnv(data_v=data_v, action_scheme_id=action_scheme_id, obs_dim=obs_dim,
                 auto_follow=auto_follow, max_ep_len=max_ep_len, trainning_set=trainning_set)
if num_stack > 1:
    env = FrameStack(env, num_stack, jump=3, model=model)

env = EnvWrapper(env, delay_len=delay_len, target_scale=target_scale, score_scale=score_scale,
                 profit_scale=profit_scale, action_punish=ap, target_clip=target_clip, burn_in=burn_in)

ppo(lambda: env, data_v=data_v)
