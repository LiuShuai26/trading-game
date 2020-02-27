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

deterministic = False
num_episodes = 100
max_ep_len = 1000

fpath = "/home/shuai/trading-game/spinningup/data/ppo-target-256-lowlr/ppo-target-256-lowlr_s0"

fname = osp.join(fpath, 'tf1_save')
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

env = TradingEnv()

logger = EpochLogger()
o, r, d, ep_ret, ep_len, n = env.reset(start_day=1, skip_step=10000, analyse=True), 0, False, 0, 0, 0
while n < num_episodes:

    a = get_action(o)
    o, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    if d or (ep_len == max_ep_len):
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
        all_data = env.all_data
        all_data_df = pd.DataFrame(all_data)
        print(all_data_df.tail())
        all_data_df.to_csv("/home/shuai/day1-test-target.csv", index=False)
        print("data save!")
        break
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        n += 1

logger.log_tabular('EpRet', with_min_and_max=True)
logger.log_tabular('EpLen', average_only=True)
logger.dump_tabular()
