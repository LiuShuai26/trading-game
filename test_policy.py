import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
from env import TradingEnv
import pandas as pd
import numpy as np

data_len = [
    225016, 225018, 225018, 225018, 225018, 225017, 225018, 225016, 225014, 225016, 225016, 225018, 225018, 225015,
    225018, 225016, 177490, 225016, 225018, 225016, 225016, 225016, 225018, 225016, 225018, 225018, 225016, 225016,
    225016, 225018, 225018, 225016, 225016, 225018, 225016, 225016, 225018, 225016, 225016, 225015, 225016, 225016,
    225016, 225016, 192623, 225018, 225018, 225016, 225016, 225016, 225016, 225018, 225016, 225018, 225016, 225016,
    225016, 225016, 99006, 225016, 225018, 99010
]

# -----------setting------------
deterministic = False
num_episodes = 10
max_ep_len = 3000

save_data = False
if num_episodes == 1:
    save_data = True

test_all = True

exp_name = "ppo-trading-dayall-clip3-ds=True-fs=1-ts=1-ss=1-ap=0.5dl=30clip=3-pilr=5e-05-vlr=1e-05"
# exp_name = "ppo-baseset-c12-ap0.5-fs=1-ss=1.0-ap=0.5"
fpath = "/home/shuai/trading-game/spinningup/data/" + exp_name + "/" + exp_name + "_s0"
fname = osp.join(fpath, 'tf1_save1')
# 6000: 137
# 5000: 133
# 4000: 133
# 3000: 137
# 2000: 141
# 1000: 133

# -----------setting------------

print('\n\nLoading from %s.\n\n ' % fname)
# load the things!
sess = tf.Session()
model = restore_tf_graph(sess, fname)

# get the correct op for executing actions
if deterministic and 'mu' in model.keys():
    # 'deterministic' is only a valid option for SAC policies
    print('Using deterministic action op.')
    action_op = model['mu']
else:
    print('Using default action op.')
    action_op = model['pi']

# make function for producing an action given a single state
get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]


def baseline069(obs):
    if obs[24] > obs[25]:
        action = 6
    elif obs[24] < obs[25]:
        action = 9
    else:
        action = 0
    return action


def random_test(num_episodes):
    start_day = np.random.randint(1, 63, num_episodes)
    skip_step = []
    for start in start_day:
        data_len_index = start - 1
        skip_step.append(int(np.random.randint(0, data_len[data_len_index] - max_ep_len, 1)[0]))

    print(start_day)
    print(skip_step)
    print("save data:", save_data)

    env = TradingEnv()

    logger = EpochLogger()

    n = 0
    for start, skip in zip(start_day, skip_step):
        o, r, d, ep_ret, ep_len = env.reset(start_day=start, start_skip=skip, analyse=True), 0, False, 0, 0
        # o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

        ep_target_bias, ep_apnum, ep_score = 0, 0, 0
        while True:
            a = get_action(o)
            # a = baseline069(o)

            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            ep_target_bias += info["target_bias"]
            ep_apnum += info["ap_num"]
            ep_score += info["one_step_score"]

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpTarget_bias=ep_target_bias, Target_bias_per_step=ep_target_bias / ep_len,
                             EpApNum=ep_apnum, EpScore=ep_score,
                             EpLen=ep_len)
                # print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
                n += 1
                if save_data:
                    all_data = env.all_data
                    all_data_df = pd.DataFrame(all_data)
                    print(all_data_df.tail())
                    all_data_df.to_csv("/home/shuai/day1-test-basesetap0.5.csv", index=True)
                    print("data save!")
                print('Agent1 : %s [%d/%d]' % (str(round(1.0 * n / num_episodes * 100)) + '%', n, num_episodes),
                      end='\r')
                break

    logger.log_tabular('EpScore', with_min_and_max=True)
    logger.log_tabular('EpTarget_bias', with_min_and_max=True)
    logger.log_tabular('Target_bias_per_step', average_only=True)
    logger.log_tabular('EpApNum', average_only=True)
    logger.log_tabular('EpRet', average_only=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    logger = EpochLogger()

    n = 0
    for start, skip in zip(start_day, skip_step):
        o, r, d, ep_ret, ep_len = env.reset(start_day=start, start_skip=skip, analyse=True), 0, False, 0, 0
        # o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

        ep_target_bias, ep_apnum, ep_score = 0, 0, 0
        while True:
            # a = get_action(o)
            a = baseline069(o)

            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            ep_target_bias += info["target_bias"]
            ep_apnum += info["ap_num"]
            ep_score += info["one_step_score"]

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpTarget_bias=ep_target_bias, Target_bias_per_step=ep_target_bias / ep_len,
                             EpApNum=ep_apnum, EpScore=ep_score,
                             EpLen=ep_len)
                # print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
                n += 1
                if save_data:
                    all_data = env.all_data
                    all_data_df = pd.DataFrame(all_data)
                    print(all_data_df.tail())
                    all_data_df.to_csv("/home/shuai/day1-test-069.csv", index=True)
                    print("data save!")
                print('Agent069 : %s [%d/%d]' % (str(round(1.0 * n / num_episodes * 100)) + '%', n, num_episodes),
                      end='\r')
                break

    logger.log_tabular('EpScore', with_min_and_max=True)
    logger.log_tabular('EpTarget_bias', with_min_and_max=True)
    logger.log_tabular('Target_bias_per_step', average_only=True)
    logger.log_tabular('EpApNum', average_only=True)
    logger.log_tabular('EpRet', average_only=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def full_test():
    env = TradingEnv()
    logger = EpochLogger()
    for start in range(1, 63):
        o, r, d, ep_ret, ep_len = env.reset(start_day=start, start_skip=0, analyse=False), 0, False, 0, 0
        ep_target_bias, ep_apnum = 0, 0
        while True:
            a = get_action(o)
            # a = baseline069(o)

            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            ep_target_bias += info["target_bias"]
            ep_apnum += info["ap_num"]

            if d:
                logger.store(EpRet=ep_ret, Target_bias_per_step=ep_target_bias/ep_len,
                             ApNum_per_step=ep_apnum/ep_len,
                             EpScore=info["score"],
                             EpLen=ep_len)
                print("Day", start, "Profit:", info["profit"], "Score:", info["score"])
                break

    logger.log_tabular('EpScore', with_min_and_max=True)
    logger.log_tabular('Target_bias_per_step', with_min_and_max=True)
    logger.log_tabular('ApNum_per_step', average_only=True)
    logger.log_tabular('EpRet', average_only=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if test_all:
    full_test()
else:
    random_test(num_episodes)
