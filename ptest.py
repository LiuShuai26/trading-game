import os.path as osp
import tensorflow as tf
import sys
sys.path.append("/home/shuai/trading-game/spinningup/")
from spinup.utils.logx import restore_tf_graph
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='day32rld96_35-model=mlp[600,800,600]-obs_dim=26-as15-auto_follow=0-burn_in-3000-fs=1-ts=1-ss=1.5-ap=0.5dl=30clip=5-lr=4e-05')
parser.add_argument('--model', type=str, default='tf1_save113.400')
args = parser.parse_args()


# ----------------- Setting --------------------------------------
num_cpu = 1
num_stack = 1

assert num_cpu < 63, "num_cpu should < 63"
mpi_fork(num_cpu, bind_to_core=False, cpu_set="")  # run parallel code with mpi

fpath = "/home/shuai/trading-game/spinningup/data/" + args.exp_name + '/' + args.exp_name + '_s0/'
fname = osp.join(fpath, args.model)
# --------------------------------------------------------------------------

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
from trading_env import TradingEnv, FrameStack
from wrapper import EnvWrapper
env = TradingEnv(action_scheme_id=15, obs_dim=26)
logger = EpochLogger()
if num_stack > 1:
    env = FrameStack(env, num_stack)

for start in range(proc_id()+1, 63, num_cpu):
    o, r, d, ep_ret, ep_len = env.reset(start_day=start, start_skip=0), 0, False, 0, 0
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
