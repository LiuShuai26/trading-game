import os.path as osp
import tensorflow as tf
import sys

ROOT = osp.abspath(osp.dirname(__file__))
sys.path.append(ROOT + "/spinningup/")
import spinup.algos.tf1.ppo.core as core
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()
parser.add_argument('--data_v', type=str, default='r19')
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[600, 800, 600])
parser.add_argument('--num_cpu', type=int, default=30)
parser.add_argument('--num_stack', type=int, default=1)
parser.add_argument('--test_days', type=int, default=30)
parser.add_argument('--action_scheme', type=int, default=15)
parser.add_argument('--obs_dim', type=int, default=26)
parser.add_argument('--exp_name', type=str, default='/spinningup/data/')
parser.add_argument('--weights', type=str, default='tf1_save221.0M108.125p')
args = parser.parse_args()

assert args.num_cpu < 63, "num_cpu should < 63"
mpi_fork(args.num_cpu)  # run parallel code with mpi

# --------------------------------------------------------------------------
from spinup import EpochLogger
from game_env.new_env import TradingEnv, FrameStack

env = TradingEnv(data_v=args.data_v, action_scheme_id=args.action_scheme, obs_dim=args.obs_dim)
logger = EpochLogger()
if args.num_stack > 1:
    env = FrameStack(env, args.num_stack)
# env = EnvWrapper(env)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape

ac_kwargs = dict(hidden_sizes=args.hidden_sizes)
# Share information about action space with policy architecture
ac_kwargs['action_space'] = env.action_space

# Inputs to computation graph
x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)

actor_critic = core.mlp_actor_critic
# Main outputs from computation graph
pi, logp, logp_pi, h, v = actor_critic(x_ph, a_ph, **ac_kwargs)

get_action = lambda x: sess.run(pi, feed_dict={x_ph: x[None, :]})[0]

sess = tf.Session()
# restore
fpath = ROOT + "/spinningup/data/"
saver = tf.train.Saver()
saver.restore(sess, fpath + '/' + args.weights + '/model.ckpt')
print("******", args.weights, "restored! ******")
# restore end

for start in range(proc_id() + 1, args.test_days + 1, args.num_cpu):
    o, r, d, ep_ret, ep_len = env.reset(start_day=start), 0, False, 0, 0
    ep_target_bias, ep_apnum = 0, 0
    while True:
        a = get_action(o)
        o, r, d, info = env.step(a)
        ep_len += 1
        ep_target_bias += info["target_bias"]

        # if d or ep_len == 3000:  # for fast debug
        if d:
            logger.store(Target_bias_per_step=ep_target_bias / ep_len,
                         EpScore=info["score"],
                         EpLen=ep_len)
            print("Day", start, "len:", ep_len, "Profit:", info["profit"], "Score:", info["score"])
            break

logger.log_tabular('EpScore', with_min_and_max=True)
logger.log_tabular('Target_bias_per_step', with_min_and_max=True)
logger.log_tabular('EpLen', average_only=True)
logger.dump_tabular()

os._exit(8)
