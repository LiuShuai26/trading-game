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

deterministic = False
num_episodes = 1
max_ep_len = 3000

# exp_name = "ppo-delayed_target-m343-newaction"
exp_name = "ppo-m343-b36000-l3000-dt_30-c12-set4"


fpath = "/home/shuai/trading-game/spinningup/data/" + exp_name + "/" + exp_name + "_s0"

fname = osp.join(fpath, 'tf1_save7600')
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
        action = 7
    elif obs[24] < obs[25]:
        action = 8
    else:
        action = 0
    return action


start_day = np.random.randint(1, 63, num_episodes)
skip_step = []
for start in start_day:
    data_len_index = start - 1
    skip_step.append(int(np.random.randint(0, data_len[data_len_index] - max_ep_len, 1)[0]))

env = TradingEnv()

logger = EpochLogger()

for start, skip in zip(start_day, skip_step):
    o, r, d, ep_ret, ep_len, n = env.reset(start_day=start, skip_step=skip, analyse=True), 0, False, 0, 0, 0
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
        ep_score += info["score"]

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpTarget_bias=ep_target_bias, EpApNum=ep_apnum, EpScore=ep_score, EpLen=ep_len)
            # print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            n += 1
            all_data = env.all_data
            all_data_df = pd.DataFrame(all_data)
            print(all_data_df.tail())
            all_data_df.to_csv("/home/shuai/day1-test-set4.csv", index=True)
            print("data save!")
            # exit()
            break


logger.log_tabular('EpRet', with_min_and_max=True)
logger.log_tabular('EpTarget_bias', with_min_and_max=True)
logger.log_tabular('EpApNum', with_min_and_max=True)
logger.log_tabular('EpScore', with_min_and_max=True)
logger.log_tabular('EpLen', average_only=True)
logger.dump_tabular()


logger = EpochLogger()

for start, skip in zip(start_day, skip_step):
    o, r, d, ep_ret, ep_len, n = env.reset(start_day=start, skip_step=skip, analyse=True), 0, False, 0, 0, 0
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
        ep_score += info["score"]

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpTarget_bias=ep_target_bias, EpApNum=ep_apnum, EpScore=ep_score, EpLen=ep_len)
            # print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            n += 1
            all_data = env.all_data
            all_data_df = pd.DataFrame(all_data)
            print(all_data_df.tail())
            all_data_df.to_csv("/home/shuai/day1-test-078.csv", index=True)
            print("data save!")
            break

logger.log_tabular('EpRet', with_min_and_max=True)
logger.log_tabular('EpTarget_bias', with_min_and_max=True)
logger.log_tabular('EpApNum', with_min_and_max=True)
logger.log_tabular('EpScore', with_min_and_max=True)
logger.log_tabular('EpLen', average_only=True)
logger.dump_tabular()
