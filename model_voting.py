import os.path as osp
import tensorflow as tf
import sys

sys.path.append(sys.path[0] + '/spinningup')
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = ""

fpath = "/home/shuai/gameR12/trading-game/spinningup/data/"

# fname = osp.join(fpath, 'tf1_save190.080')
#
# print('\n\nLoading from %s.\n\n ' % fname)
# # load the things!
# sess1 = tf.Session()
# model1 = restore_tf_graph(sess1, fname)
# print('Using default action op.')
# action_op1 = model1['pi']
#
# # make function for producing an action given a single state
# get_action1 = lambda x: sess1.run(action_op1, feed_dict={model1['x']: x[None, :]})[0]
#
#
# fname2 = osp.join(fpath, 'tf1_save420.120')
#
# print('\n\nLoading from %s.\n\n ' % fname2)
# # load the things!
# # sess2 = tf.Session()
# model2 = restore_tf_graph(sess1, fname2)
# print('Using default action op.')
# action_op2 = model2['pi']
#
# # make function for producing an action given a single state
# get_action2 = lambda x: sess1.run(action_op2, feed_dict={model2['x']: x[None, :]})[0]


# model_names = ['tf1_save113.400', 'tf1_save136.080', 'tf1_save142.560',
#                'tf1_save166.320', 'tf1_save169.560', 'tf1_save173.880', 'tf1_save190.080', 'tf1_save213.840',
#                'tf1_save216.000', 'tf1_save420.120', 'tf1_save459.000', 'tf1_save87.480']
model_names = ['tf1_save190.080', 'tf1_save166.320', 'tf1_save213.840', 'tf1_save420.120', 'tf1_save459.000']   # 'tf1_save136.080'


models = []
action_ops = []
get_actions = []

sess = tf.Session()

for i, model_name in enumerate(model_names):
    fname = osp.join(fpath, model_name)
    print('\n\nLoading from %s.\n\n ' % fname)

    # load the things!
    models.append(restore_tf_graph(sess, fname))
    action_ops.append(models[i]['pi'])
    get_actions.append(lambda x: sess.run(action_ops[i], feed_dict={models[i]['x']: x[None, :]})[0])

# print(sessions)
# print(models)
# print(get_actions)
# --------------------------------------------------------------------------
from spinup import EpochLogger
from trading_env import TradingEnv, FrameStack
from wrapper import EnvWrapper

env = TradingEnv(action_scheme_id=15, obs_dim=26)
logger = EpochLogger()

# env = EnvWrapper(env)

for start in range(1, 62 + 1):
    o, r, d, ep_ret, ep_len = env.reset(start_day=start, start_skip=0), 0, False, 0, 0
    ep_target_bias, ep_apnum = 0, 0
    while True:
        # a1 = get_action1(o)
        actions = []
        for get_action in get_actions:
            actions.append(get_action(o))
        a = Counter(actions).most_common(1)[0][0]
        # if 0 not in a:
        #     print("Step:", ep_len, "actions:", a)
        o, r, d, info = env.step(a)
        ep_len += 1
        ep_target_bias += info["target_bias"]

        if d:
            logger.store(Target_bias_per_step=ep_target_bias / ep_len,
                         ApNum_per_step=0,
                         EpScore=info["score"],
                         EpLen=ep_len)
            print("Day", start, "len:", ep_len, "Profit:", info["profit"], "Score:", info["score"])
            break

logger.log_tabular('EpScore', with_min_and_max=True)
logger.log_tabular('Target_bias_per_step', with_min_and_max=True)
logger.log_tabular('ApNum_per_step', average_only=True)
logger.log_tabular('EpLen', average_only=True)
logger.dump_tabular()

os._exit(8)
