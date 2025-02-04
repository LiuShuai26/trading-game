import os.path as osp
import tensorflow as tf
import sys

ROOT = osp.abspath(osp.dirname(__file__))
sys.path.append(ROOT+"/spinningup/")
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()
parser.add_argument('--num_cpu', type=int, default=30)
parser.add_argument('--num_stack', type=int, default=1)
parser.add_argument('--test_days', type=int, default=30)
parser.add_argument('--actions', type=int, default=15)
parser.add_argument('--obs_dim', type=int, default=26)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--model', type=str, default='tf1_save190.080')
args = parser.parse_args()

assert args.num_cpu < 63, "num_cpu should < 63"
mpi_fork(args.num_cpu)  # run parallel code with mpi

if args.exp_name is not None:
    fpath = ROOT + "/spinningup/data/" + args.exp_name + '/' + args.exp_name + '_s0/'
else:
    fpath = ROOT + "/spinningup/data/"
fname = osp.join(fpath, args.model)

print('\n\nLoading from %s.\n\n ' % fname)
# load the things!
sess = tf.Session()
model = restore_tf_graph(sess, fname)
print('Using default action op.')
action_op = model['pi']

# make function for producing an action given a single state
get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

# --------------------------------------------------------------------------
from spinup import EpochLogger
from game_env.new_env import TradingEnv, FrameStack

env = TradingEnv(data_v="r12", action_scheme_id=args.actions, obs_dim=args.obs_dim)

logger = EpochLogger()
if args.num_stack > 1:
    env = FrameStack(env, args.num_stack)
# env = EnvWrapper(env)

for start in range(proc_id() + 91, args.test_days + 91, args.num_cpu):
    o, r, d, ep_ret, ep_len = env.reset(start_day=start), 0, False, 0, 0
    ep_target_bias, ep_apnum = 0, 0
    while True:
        a = get_action(o)
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
