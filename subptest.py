import os.path as osp
import tensorflow as tf
import sys
sys.path.append("/home/shuai/trading-game/spinningup/")
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpu', type=int, default=4)
parser.add_argument('--num_stack', type=int, default=1)
parser.add_argument('--start', type=int, default=1)
parser.add_argument('--actions', type=int, default=15)
parser.add_argument('--obs_dim', type=int, default=26)
parser.add_argument('--exp_name', type=str, default='ppo-trainning_set=54-model=mlp-obs_dim=26-as15-auto_follow=0-burn_in-3000-fs=1-ts=1-ss=1.5-ap=0.4dl=30clip=5-lr=4e-05')
parser.add_argument('--model', type=str, default='last_tf1_save')
args = parser.parse_args()

assert args.num_cpu < 63, "num_cpu should < 63"
assert args.exp_name is not 'ppo-trainning', "you are using default exp_name!"

fpath = "/home/shuai/trading-game/spinningup/data/" + args.exp_name + '/' + args.exp_name + '_s0/'
# fpath = "/home/shuai/trading-game/spinningup/data/"
fname = osp.join(fpath, args.model)

# print('\n\nLoading from %s.\n\n ' % fname)
# load the things!
sess = tf.Session()
model = restore_tf_graph(sess, fname)
# print('Using default action op.')
action_op = model['pi']

# make function for producing an action given a single state
get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

# --------------------------------------------------------------------------

from trading_env import TradingEnv, FrameStack
from wrapper import EnvWrapper

env = TradingEnv(action_scheme_id=args.actions, obs_dim=args.obs_dim)

if args.num_stack > 1:
    env = FrameStack(env, args.num_stack)
# env = EnvWrapper(env)

o, r, d, ep_ret, ep_len = env.reset(start_day=args.start, start_skip=0), 0, False, 0, 0
ep_target_bias, ep_apnum = 0, 0
while True:
    a = get_action(o)
    o, r, d, info = env.step(a)
    ep_len += 1
    ep_target_bias += info["target_bias"]

    if d:
        print(info["score"])
        os._exit(8)
