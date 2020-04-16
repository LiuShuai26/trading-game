import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.tf1.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import sys
import datetime


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=3000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=5e7, exp_name='exp', summary_dir="/home/shuai/tb",
        start_day=None, start_skip=None, duration=None, burn_in=0):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Tensorflow Summary Ops
    def build_summaries():
        summaries = []
        EpRet = tf.Variable(0.)
        EpRet_target_bias = tf.Variable(0.)
        EpRet_score = tf.Variable(0.)
        EpApNum = tf.Variable(0.)
        EpTarget_bias = tf.Variable(0.)
        Target_bias_per_step = tf.Variable(0.)
        EpScore = tf.Variable(0.)

        Action0 = tf.Variable(0.)
        Action1 = tf.Variable(0.)
        Action2 = tf.Variable(0.)
        Action3 = tf.Variable(0.)
        Action4 = tf.Variable(0.)
        Action5 = tf.Variable(0.)
        Action6 = tf.Variable(0.)
        Action7 = tf.Variable(0.)
        Action8 = tf.Variable(0.)
        Action9 = tf.Variable(0.)
        Action10 = tf.Variable(0.)
        Action11 = tf.Variable(0.)
        Action12 = tf.Variable(0.)
        Action13 = tf.Variable(0.)
        Action14 = tf.Variable(0.)

        summaries.append(tf.summary.scalar("Reward", EpRet))
        summaries.append(tf.summary.scalar("EpRet_target_bias", EpRet_target_bias))
        summaries.append(tf.summary.scalar("EpRet_score", EpRet_score))
        summaries.append(tf.summary.scalar("EpApNum", EpApNum))
        summaries.append(tf.summary.scalar("EpTarget_bias", EpTarget_bias))
        summaries.append(tf.summary.scalar("Target_bias_per_step", Target_bias_per_step))
        summaries.append(tf.summary.scalar("EpScore", EpScore))

        summaries.append(tf.summary.scalar("Action0", Action0))
        summaries.append(tf.summary.scalar("Action1", Action1))
        summaries.append(tf.summary.scalar("Action2", Action2))
        summaries.append(tf.summary.scalar("Action3", Action3))
        summaries.append(tf.summary.scalar("Action4", Action4))
        summaries.append(tf.summary.scalar("Action5", Action5))
        summaries.append(tf.summary.scalar("Action6", Action6))
        summaries.append(tf.summary.scalar("Action7", Action7))
        summaries.append(tf.summary.scalar("Action8", Action8))
        summaries.append(tf.summary.scalar("Action9", Action9))
        summaries.append(tf.summary.scalar("Action10", Action10))
        summaries.append(tf.summary.scalar("Action11", Action11))
        summaries.append(tf.summary.scalar("Action12", Action12))
        summaries.append(tf.summary.scalar("Action13", Action13))
        summaries.append(tf.summary.scalar("Action14", Action14))

        test_ops = tf.summary.merge(summaries)
        test_vars = [EpRet, EpRet_target_bias, EpRet_score, EpApNum, EpTarget_bias, Target_bias_per_step, EpScore]
        test_vars += [Action0, Action1, Action2, Action3, Action4, Action5, Action6, Action7, Action8, Action9,
                      Action10, Action11, Action12, Action13, Action14]
        return test_ops, test_vars

    # Set up summary Ops
    test_ops, test_vars = build_summaries()

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.03
    config.inter_op_parallelism_threads = 1
    config.intra_op_parallelism_threads = 1
    sess = tf.Session(config=config)
    # sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    if proc_id() == 0:
        writer = tf.summary.FileWriter(
            summary_dir + "/" + str(datetime.datetime.now()) + "-" + exp_name, sess.graph)
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(start_day=start_day, start_skip=start_skip, duration=duration,
                                        burn_in=burn_in), 0, False, 0, 0
    ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_apnum = 0, 0, 0, 0, 0

    min_score = 150
    max_saved_steps = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})

            o2, r, d, info = env.step(a[0])
            ep_ret += r
            ep_len += 1
            ep_target_bias += info["target_bias"]
            ep_reward_target_bias += info["reward_target_bias"]
            ep_score += info["one_step_score"]
            ep_reward_score += info["reward_score"]
            ep_apnum += info["ap_num"]

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(AverageEpRet=ep_ret,
                                 EpRet_target_bias=ep_reward_target_bias,
                                 EpRet_score=ep_reward_score,
                                 EpApNum=ep_apnum,
                                 EpTarget_bias=ep_target_bias,
                                 Target_bias_per_step=ep_target_bias / ep_len,
                                 EpScore=ep_score,
                                 EpLen=ep_len)

                    logger.store(Action0=env.act_sta[0],
                                 Action1=env.act_sta[1],
                                 Action2=env.act_sta[2],
                                 Action3=env.act_sta[3],
                                 Action4=env.act_sta[4],
                                 Action5=env.act_sta[5],
                                 Action6=env.act_sta[6],
                                 Action7=env.act_sta[7],
                                 Action8=env.act_sta[8],
                                 Action9=env.act_sta[9],
                                 Action10=env.act_sta[10],
                                 Action11=env.act_sta[11],
                                 Action12=env.act_sta[12],
                                 Action13=env.act_sta[13],
                                 Action14=env.act_sta[14],
                                 )

                o, ep_ret, ep_len = env.reset(start_day=start_day, start_skip=start_skip, duration=duration,
                                              burn_in=burn_in), 0, 0
                ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_apnum = 0, 0, 0, 0, 0

        total_steps = (epoch + 1) * steps_per_epoch

        # tensorboard writting
        tb_ep_ret = logger.get_stats('AverageEpRet')[0]
        tb_ret_target_bias = logger.get_stats('EpRet_target_bias')[0]
        tb_ret_score = logger.get_stats('EpRet_score')[0]
        tb_apnum = logger.get_stats('EpApNum')[0]
        tb_target_bias = logger.get_stats('EpTarget_bias')[0]
        tb_target_bias_per_step = logger.get_stats('Target_bias_per_step')[0]
        tb_score = logger.get_stats('EpScore')[0]

        tb_action0 = logger.get_stats('Action0')[0]
        tb_action1 = logger.get_stats('Action1')[0]
        tb_action2 = logger.get_stats('Action2')[0]
        tb_action3 = logger.get_stats('Action3')[0]
        tb_action4 = logger.get_stats('Action4')[0]
        tb_action5 = logger.get_stats('Action5')[0]
        tb_action6 = logger.get_stats('Action6')[0]
        tb_action7 = logger.get_stats('Action7')[0]
        tb_action8 = logger.get_stats('Action8')[0]
        tb_action9 = logger.get_stats('Action9')[0]
        tb_action10 = logger.get_stats('Action10')[0]
        tb_action11 = logger.get_stats('Action11')[0]
        tb_action12 = logger.get_stats('Action12')[0]
        tb_action13 = logger.get_stats('Action13')[0]
        tb_action14 = logger.get_stats('Action14')[0]

        if proc_id() == 0:
            summary_str = sess.run(test_ops, feed_dict={
                test_vars[0]: tb_ep_ret,
                test_vars[1]: tb_ret_target_bias,
                test_vars[2]: tb_ret_score,
                test_vars[3]: tb_apnum,
                test_vars[4]: tb_target_bias,
                test_vars[5]: tb_target_bias_per_step,
                test_vars[6]: tb_score,
                test_vars[7]: tb_action0,
                test_vars[8]: tb_action1,
                test_vars[9]: tb_action2,
                test_vars[10]: tb_action3,
                test_vars[11]: tb_action4,
                test_vars[12]: tb_action5,
                test_vars[13]: tb_action6,
                test_vars[14]: tb_action7,
                test_vars[15]: tb_action8,
                test_vars[16]: tb_action9,
                test_vars[17]: tb_action10,
                test_vars[18]: tb_action11,
                test_vars[19]: tb_action12,
                test_vars[20]: tb_action13,
                test_vars[21]: tb_action14,
            })
            writer.add_summary(summary_str, total_steps)
            writer.flush()

        # Save model
        # save model every save_freq(50M) steps
        if (total_steps // save_freq > max_saved_steps) or (epoch == epochs - 1):
            max_saved_steps = total_steps // save_freq
            logger.save_state({'env': env}, step=total_steps, score=tb_score)
        # save model if lower than the min_score. min_score start from 150.
        if tb_score < min_score:
            logger.save_state({'env': env}, step=total_steps, score=tb_score)
            min_score = tb_score

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('AverageEpRet', tb_ep_ret)
        logger.log_tabular('EpRet_target_bias', tb_ret_target_bias)
        logger.log_tabular('EpRet_score', tb_ret_score)
        logger.log_tabular('EpApNum', tb_apnum)
        logger.log_tabular('EpTarget_bias', with_min_and_max=True)
        logger.log_tabular('Target_bias_per_step', tb_target_bias_per_step)
        logger.log_tabular('EpScore', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('EnvInteractsSpeed', ((epoch + 1) * steps_per_epoch) / (time.time() - start_time))
        logger.log_tabular('ExpName', exp_name)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Trading')
    parser.add_argument('--gamma', type=float, default=0.998)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--steps', type=int, default=96000)
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--target_scale', type=float, default=1)
    parser.add_argument('--score_scale', type=float, default=1)
    parser.add_argument('--ap', type=float, default=0.5)
    parser.add_argument('--exp_name', type=str, default='ppo-trading')
    args = parser.parse_args()

    # start_day = 32
    # start_skip = 20000
    # duration = 10000
    start_day = None
    start_skip = None
    duration = None
    burn_in = 1000

    data_skip = True
    delay_len = 30
    target_clip = 3
    max_ep_len = 3000
    pi_lr = 4e-05
    vf_lr = 1e-4

    exp_name = args.exp_name + "dataset=" + str(start_day) + '-skip' + str(start_skip)
    exp_name += "-ds=" + str(data_skip) + "-fs=" + str(args.num_stack)
    exp_name += "-ts=" + str(args.target_scale) + "-ss=" + str(args.score_scale) + "-ap=" + str(args.ap)
    exp_name += "dl=" + str(delay_len) + "clip=" + str(target_clip) + "-pilr=" + str(pi_lr) + "-vlr=" + str(vf_lr)

    mpi_fork(args.cpu, bind_to_core=False, cpu_set="")  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    sys.path.append("/home/shuai/trading-game")
    from trading_env import TradingEnv, FrameStack, EnvWrapper

    env = TradingEnv(action_scheme_id=15)
    if args.num_stack > 1:
        env = FrameStack(env, args.num_stack)

    ppo(lambda: EnvWrapper(env, delay_len=delay_len, target_scale=args.target_scale, score_scale=args.score_scale,
                           ap=args.ap, target_clip=target_clip, env_skip=data_skip),
        actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[600, 800, 600]), gamma=args.gamma, pi_lr=pi_lr, vf_lr=vf_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs, exp_name=exp_name, start_day=start_day, start_skip=start_skip, duration=duration,
        burn_in=burn_in)
