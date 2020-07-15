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
        action = env.policy_069()
        o, r, d, info = env.step(action)
        ep_ret += r
        ep_len += 1
        ep_target_bias += info["target_bias"]
        ep_reward_target_bias += info["reward_target_bias"]
        ep_reward_score += info["reward_score"]
        ep_reward_ap += info["reward_ap"]
        ep_score += info["one_step_score"]

        # print("info score:", info["score"], "ep score:", ep_score)
        print("one step score:", info["one_step_score"], "one step profit:", info["one_step_profit"])

        terminal = d or (ep_len == 3000)
        if terminal:
            print("len:", ep_len, "info score:", info["score"], "ep score:", ep_score)
            print("episode done.")
            o = env.reset()
            ep_ret, ep_len = 0, 0
            ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_reward_ap = 0, 0, 0, 0, 0

            test()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_v', type=str, default='r19')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[600, 800, 600])
    parser.add_argument('--gamma', type=float, default=0.998)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=72000)
    parser.add_argument('--epochs', type=int, default=3000000)
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--target_scale', type=float, default=1)
    parser.add_argument('--score_scale', type=float, default=0)
    parser.add_argument('--profit_scale', type=float, default=1)
    parser.add_argument('--ap', type=float, default=0.4)
    parser.add_argument('--burn_in', type=int, default=3000)
    parser.add_argument('--delay_len', type=int, default=30)
    parser.add_argument('--target_clip', type=int, default=4)
    parser.add_argument('--auto_follow', type=int, default=0)
    parser.add_argument('--action_scheme', type=int, default=15)
    parser.add_argument('--obs_dim', type=int, default=26)
    parser.add_argument('--max_ep_len', type=int, default=3000)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--exp_name', type=str, default='fix')
    parser.add_argument('--restore_model', type=str, default="")
    args = parser.parse_args()

    if args.data_v == "r19":
        trainning_set = 90
    else:
        trainning_set = 50

    from game_env.new_env import TradingEnv, FrameStack

    env = TradingEnv(data_v=args.data_v, action_scheme_id=args.action_scheme, obs_dim=args.obs_dim,
                     auto_follow=args.auto_follow, max_ep_len=args.max_ep_len, delay_len=args.delay_len,
                     target_scale=args.target_scale, score_scale=args.score_scale, profit_scale=args.profit_scale,
                     action_punish=args.ap, target_clip=args.target_clip, burn_in=args.burn_in)

    if args.num_stack > 1:
        env = FrameStack(env, args.num_stack, jump=3, model=args.model)

    ppo(lambda: env, data_v=args.data_v)
