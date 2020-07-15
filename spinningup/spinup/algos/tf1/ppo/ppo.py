import numpy as np
import tensorflow as tf
import time
import sys
import os
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))))
sys.path.append(ROOT)
sys.path.append(ROOT + '/spinningup')
from spinup.user_config import DEFAULT_DATA_DIR
import spinup.algos.tf1.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
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
        self.adv_buf = (self.adv_buf - adv_mean)  # / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


def ppo(env_fn, data_v, actor_critic=core.mlp_actor_critic, alpha=0.0, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, lr=3e-4, ap=0.4,
        train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=3000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=25e6, restore_model="tf1_save",
        exp_name='exp', summary_dir=ROOT + "/tb"):
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

        lr (float): Learning rate.

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
    pi, logp, logp_pi, h, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    learning_rate = tf.placeholder(tf.float32, shape=[])

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, learning_rate]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi, h]

    # Tensorflow Summary Ops
    def build_summaries():
        summaries = []

        EpRet = tf.Variable(0.)
        EpRet_target_bias = tf.Variable(0.)
        EpRet_score = tf.Variable(0.)
        EpRet_profit = tf.Variable(0.)
        EpRet_ap = tf.Variable(0.)
        EpTarget_bias = tf.Variable(0.)
        EpTarget_bias_per_step = tf.Variable(0.)
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
        Action15 = tf.Variable(0.)
        Action16 = tf.Variable(0.)

        VVals = tf.Variable(0.)
        LossPi = tf.Variable(0.)
        LossV = tf.Variable(0.)
        DeltaLossPi = tf.Variable(0.)
        DeltaLossV = tf.Variable(0.)
        Entropy = tf.Variable(0.)
        KL = tf.Variable(0.)
        ClipFrac = tf.Variable(0.)
        lr = tf.Variable(0.)
        ap = tf.Variable(0.)

        summaries.append(tf.summary.scalar("Reward", EpRet))
        summaries.append(tf.summary.scalar("EpRet_target_bias", EpRet_target_bias))
        summaries.append(tf.summary.scalar("EpRet_score", EpRet_score))
        summaries.append(tf.summary.scalar("EpRet_profit", EpRet_profit))
        summaries.append(tf.summary.scalar("EpRet_ap", EpRet_ap))
        summaries.append(tf.summary.scalar("EpTarget_bias", EpTarget_bias))
        summaries.append(tf.summary.scalar("EpTarget_bias_per_step", EpTarget_bias_per_step))
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
        summaries.append(tf.summary.scalar("Action15", Action15))
        summaries.append(tf.summary.scalar("Action16", Action16))

        summaries.append(tf.summary.scalar("VVals", VVals))
        summaries.append(tf.summary.scalar("LossPi", LossPi))
        summaries.append(tf.summary.scalar("LossV", LossV))
        summaries.append(tf.summary.scalar("DeltaLossPi", DeltaLossPi))
        summaries.append(tf.summary.scalar("DeltaLossV", DeltaLossV))
        summaries.append(tf.summary.scalar("Entropy", Entropy))
        summaries.append(tf.summary.scalar("KL", KL))
        summaries.append(tf.summary.scalar("ClipFrac", ClipFrac))
        summaries.append(tf.summary.scalar("lr", lr))
        summaries.append(tf.summary.scalar("ap", ap))

        test_summaries = []

        TestRet = tf.Variable(0.)
        TestRet_target_bias = tf.Variable(0.)
        TestRet_score = tf.Variable(0.)
        TestRet_ap = tf.Variable(0.)
        TestTarget_bias = tf.Variable(0.)
        TestTarget_bias_per_step = tf.Variable(0.)
        TestScore = tf.Variable(0.)

        test_summaries.append(tf.summary.scalar("TestReward", TestRet))
        test_summaries.append(tf.summary.scalar("TestRet_target_bias", TestRet_target_bias))
        test_summaries.append(tf.summary.scalar("TestRet_score", TestRet_score))
        test_summaries.append(tf.summary.scalar("TestRet_ap", TestRet_ap))
        test_summaries.append(tf.summary.scalar("TestTarget_bias", TestTarget_bias))
        test_summaries.append(tf.summary.scalar("TestTarget_bias_per_step", TestTarget_bias_per_step))
        test_summaries.append(tf.summary.scalar("TestScore", TestScore))

        train_data_ops = tf.summary.merge(summaries)
        train_data_vars = [EpRet, EpRet_target_bias, EpRet_score, EpRet_profit, EpRet_ap, EpTarget_bias, EpTarget_bias_per_step,
                           EpScore]
        train_data_vars += [Action0, Action1, Action2, Action3, Action4, Action5, Action6, Action7, Action8, Action9,
                            Action10, Action11, Action12, Action13, Action14, Action15, Action16]
        train_data_vars += [VVals, LossPi, LossV, DeltaLossPi, DeltaLossV, Entropy, KL, ClipFrac, lr, ap]

        test_data_ops = tf.summary.merge(test_summaries)
        test_data_vars = [TestRet, TestRet_target_bias, TestRet_score, TestRet_ap, TestTarget_bias,
                          TestTarget_bias_per_step, TestScore]
        return train_data_ops, train_data_vars, test_data_ops, test_data_vars

    # Set up summary Ops
    train_data_ops, train_data_vars, test_data_ops, test_data_vars = build_summaries()

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)

    # Scheme2: SPPO NO.2: add entropy
    # min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
    # pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv) + alpha * h)

    # # Scheme3: SPPO NO.3: add entropy
    adv_logp = adv_ph - alpha * logp_old_ph
    min_adv = tf.where(adv_logp > 0, (1 + clip_ratio) * adv_logp, (1 - clip_ratio) * adv_logp)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_logp, min_adv))

    v_loss = tf.reduce_mean((ret_ph - v) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    # approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    approx_ent = tf.reduce_mean(h)
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=learning_rate).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=learning_rate).minimize(v_loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # restore
    saver = tf.train.Saver()
    if restore_model:
        saver.restore(sess, DEFAULT_DATA_DIR + '/' + restore_model + '/model.ckpt')
        print("******", restore_model, "restored! ******")
    # restore end

    # Sync params across processes
    sess.run(sync_all_params())

    if proc_id() == 0:
        writer = tf.summary.FileWriter(
            summary_dir + "/" + str(datetime.datetime.now()).replace(" ", "-") + "-" + exp_name, sess.graph)
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update(epoch):
        inputs = {k: v for k, v in zip(all_phs[:-1], buf.get())}

        inputs[all_phs[-1]] = decay_learning_rate
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
                     DeltaLossV=(v_l_new - v_l_old), lr=decay_learning_rate, ap=decay_ap)

    def test():
        get_action = lambda x: sess.run(pi, feed_dict={x_ph: x[None, :]})[0]
        if data_v == "r19":
            start_test = 91
        else:
            start_test = 51
        for start_day in range(proc_id() + start_test, 8 + start_test, 8):
            o, r, d, test_ret, test_len = env.reset(ap=decay_ap, start_day=start_day), 0, False, 0, 0
            test_target_bias, test_reward_target_bias, test_reward_score, test_reward_ap = 0, 0, 0, 0
            while True:
                a = get_action(o)
                o, r, d, info = env.step(a)
                test_ret += r
                test_len += 1
                test_target_bias += info["target_bias"]
                test_reward_target_bias += info["reward_target_bias"]
                test_reward_score += info["reward_score"]
                test_reward_ap += info["reward_ap"]

                # if d or test_len == 3000:   # for fast debug
                if d:
                    logger.store(AverageTestRet=test_ret,
                                 TestRet_target_bias=test_reward_target_bias,
                                 TestRet_score=test_reward_score,
                                 TestRet_ap=test_reward_ap,
                                 TestTarget_bias=test_target_bias,
                                 TestTarget_bias_per_step=test_target_bias / test_len,
                                 TestScore=info["score"],
                                 TestLen=test_len)
                    print("Day", start_day, "Len:", test_len, "Profit:", info["profit"], "Score:", info["score"])
                    break
        # after test we need reset the env
        o, ep_ret, ep_len = env.reset(ap=decay_ap), 0, 0

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(ap=ap), 0, False, 0, 0
    ep_target_bias, ep_reward_target_bias, ep_score, ep_reward_score, ep_reward_profit, ep_reward_ap = 0, 0, 0, 0, 0, 0

    min_score = 150

    if restore_model:
        with open(DEFAULT_DATA_DIR + '/' + restore_model + '/decayp.pickle', "rb") as pickle_in:
            decayp = pickle.load(pickle_in)
            ap = decayp['decay_ap']
            lr = decayp['decay_learning_rate']
            print("****** decay parameters restored! lr:", lr, "ap:", ap, "******")
    decay_ap = ap

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t, h_t = sess.run(get_action_ops, feed_dict={x_ph: o[np.newaxis, ...]})
            rh = r + alpha * h_t
            o2, r, d, info = env.step(a[0])
            ep_ret += r
            ep_len += 1
            ep_target_bias += info["target_bias"]
            ep_reward_target_bias += info["reward_target_bias"]
            ep_score += info["one_step_score"]
            ep_reward_score += info["reward_score"]
            ep_reward_profit += info["reward_profit"]
            ep_reward_ap += info["reward_ap"]

            # save and log
            buf.store(o, a, rh, v_t, logp_t)
            logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: o[np.newaxis, ...]})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(AverageEpRet=ep_ret,
                                 EpRet_target_bias=ep_reward_target_bias,
                                 EpRet_score=ep_reward_score,
                                 EpRet_profit=ep_reward_profit,
                                 EpRet_ap=ep_reward_ap,
                                 EpTarget_bias=ep_target_bias,
                                 EpTarget_bias_per_step=ep_target_bias / ep_len,
                                 EpScore=info['score'],
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
                                 Action15=env.act_sta[15],
                                 Action16=env.act_sta[16],
                                 )

                # if d:
                o = env.reset(ap=decay_ap)
                # env.act_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
                ep_ret, ep_reward_target_bias, ep_reward_score, ep_reward_profit, ep_reward_ap = 0, 0, 0, 0, 0
                ep_target_bias, ep_score, ep_len = 0, 0, 0

        total_steps = (epoch + 1) * steps_per_epoch

        decay_ap = ap * (0.9 ** (epoch // 35))
        decay_learning_rate = max(lr * (0.96 ** (epoch // 35)), 3e-6)

        # Perform PPO update!
        update(epoch)

        # tensorboard writting
        tb_ep_ret = logger.get_stats('AverageEpRet')[0]
        tb_ret_target_bias = logger.get_stats('EpRet_target_bias')[0]
        tb_ret_score = logger.get_stats('EpRet_score')[0]
        tb_ret_profit = logger.get_stats('EpRet_profit')[0]
        tb_ret_ap = logger.get_stats('EpRet_ap')[0]
        tb_target_bias = logger.get_stats('EpTarget_bias')[0]
        tb_target_bias_per_step = logger.get_stats('EpTarget_bias_per_step')[0]
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
        tb_action15 = logger.get_stats('Action15')[0]
        tb_action16 = logger.get_stats('Action16')[0]

        tb_vvals = logger.get_stats('VVals')[0]
        tb_losspi = logger.get_stats('LossPi')[0]
        tb_lossv = logger.get_stats('LossV')[0]
        tb_deltalosspi = logger.get_stats('DeltaLossPi')[0]
        tb_deltalossv = logger.get_stats('DeltaLossV')[0]
        tb_entropy = logger.get_stats('Entropy')[0]
        tb_kl = logger.get_stats('KL')[0]
        tb_clipfrac = logger.get_stats('ClipFrac')[0]
        tb_lr = logger.get_stats('lr')[0]
        tb_ap = logger.get_stats('ap')[0]

        if proc_id() == 0:
            summary_str = sess.run(train_data_ops, feed_dict={
                train_data_vars[0]: tb_ep_ret,
                train_data_vars[1]: tb_ret_target_bias,
                train_data_vars[2]: tb_ret_score,
                train_data_vars[3]: tb_ret_profit,
                train_data_vars[4]: tb_ret_ap,
                train_data_vars[5]: tb_target_bias,
                train_data_vars[6]: tb_target_bias_per_step,
                train_data_vars[7]: tb_score,
                train_data_vars[8]: tb_action0,
                train_data_vars[9]: tb_action1,
                train_data_vars[10]: tb_action2,
                train_data_vars[11]: tb_action3,
                train_data_vars[12]: tb_action4,
                train_data_vars[13]: tb_action5,
                train_data_vars[14]: tb_action6,
                train_data_vars[15]: tb_action7,
                train_data_vars[16]: tb_action8,
                train_data_vars[17]: tb_action9,
                train_data_vars[18]: tb_action10,
                train_data_vars[19]: tb_action11,
                train_data_vars[20]: tb_action12,
                train_data_vars[21]: tb_action13,
                train_data_vars[22]: tb_action14,
                train_data_vars[23]: tb_action15,
                train_data_vars[24]: tb_action16,
                train_data_vars[25]: tb_vvals,
                train_data_vars[26]: tb_losspi,
                train_data_vars[27]: tb_lossv,
                train_data_vars[28]: tb_deltalosspi,
                train_data_vars[29]: tb_deltalossv,
                train_data_vars[30]: tb_entropy,
                train_data_vars[31]: tb_kl,
                train_data_vars[32]: tb_clipfrac,
                train_data_vars[33]: tb_lr,
                train_data_vars[34]: tb_ap,
            })
            writer.add_summary(summary_str, total_steps)
            writer.flush()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('AverageEpRet', tb_ep_ret)
        logger.log_tabular('EpRet_target_bias', tb_ret_target_bias)
        logger.log_tabular('EpRet_score', tb_ret_score)
        logger.log_tabular('EpRet_profit', tb_ret_profit)
        logger.log_tabular('EpRet_ap', tb_ret_ap)
        logger.log_tabular('EpTarget_bias', with_min_and_max=True)
        logger.log_tabular('EpTarget_bias_per_step', tb_target_bias_per_step)
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
        logger.log_tabular('lr', average_only=True)
        logger.log_tabular('ap', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('EnvInteractsSpeed', ((epoch + 1) * steps_per_epoch) / (time.time() - start_time))
        logger.log_tabular('ExpName', exp_name)
        logger.dump_tabular()
        logger.clear_epoch_dict()

        # if True:          # for fast debug
        if (epoch + 1) % 15 == 0 and tb_target_bias_per_step < 10:
            test()

            test_ep_ret = logger.get_stats('AverageTestRet')[0]
            test_ret_target_bias = logger.get_stats('TestRet_target_bias')[0]
            test_ret_score = logger.get_stats('TestRet_score')[0]
            test_ret_ap = logger.get_stats('TestRet_ap')[0]
            test_target_bias = logger.get_stats('TestTarget_bias')[0]
            test_target_bias_per_step = logger.get_stats('TestTarget_bias_per_step')[0]
            test_score = logger.get_stats('TestScore')[0]

            if proc_id() == 0:

                # save model if lower than the min_score. min_score start from 150.
                if test_score < min_score:
                    min_score = test_score
                    subfolder = '/tf1_save' + str(total_steps // 1e6) + 'M' + str(min_score) + 'p'
                    save_path = saver.save(sess, logger_kwargs['output_dir'] + subfolder + '/model.ckpt')
                    decayp = {'decay_ap': decay_ap, 'decay_learning_rate': decay_learning_rate}
                    with open(logger_kwargs['output_dir'] + subfolder + "/decayp.pickle", "wb") as pickle_out:
                        pickle.dump(decayp, pickle_out)
                    print("Model saved in path: %s" % save_path)

                summary_str = sess.run(test_data_ops, feed_dict={
                    test_data_vars[0]: test_ep_ret,
                    test_data_vars[1]: test_ret_target_bias,
                    test_data_vars[2]: test_ret_score,
                    test_data_vars[3]: test_ret_ap,
                    test_data_vars[4]: test_target_bias,
                    test_data_vars[5]: test_target_bias_per_step,
                    test_data_vars[6]: test_score,
                })
                writer.add_summary(summary_str, total_steps)
                writer.flush()

            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('AverageTestRet', test_ep_ret)
            logger.log_tabular('TestRet_target_bias', test_ret_target_bias)
            logger.log_tabular('TestRet_score', test_ret_score)
            logger.log_tabular('TestRet_ap', test_ret_ap)
            logger.log_tabular('TestTarget_bias', test_target_bias)
            logger.log_tabular('TestTarget_bias_per_step', test_target_bias_per_step)
            logger.log_tabular('TestScore', test_score)
            logger.log_tabular('TestLen', average_only=True)
            logger.dump_tabular()


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
    parser.add_argument('--score_scale', type=float, default=1.5)
    parser.add_argument('--profit_scale', type=float, default=0)
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

    assert args.model in ['mlp', 'cnn'], "model must be mlp or cnn."
    assert args.data_v in ['r12', 'r19'], "data version must be r12 or r19."
    assert args.steps / args.cpu >= args.max_ep_len, "steps/cpu should >= max_ep_len: each cpu at least run one full episode."

    if args.data_v == "r19":
        trainning_set = 90
    else:
        trainning_set = 50

    exp_name = args.exp_name + "-dataV-" + args.data_v
    exp_name += "-trainning_set" + str(trainning_set) + "-model=" + args.model + str(args.hidden_sizes)[1:-1].replace(" ", "")
    exp_name += "-obs_dim" + str(args.obs_dim) + "-as" + str(args.action_scheme)
    exp_name += "-auto_follow" + str(args.auto_follow) + "-max_ep_len" + str(args.max_ep_len) + "-burn_in" + str(args.burn_in)
    exp_name += "-fs" + str(args.num_stack)
    exp_name += "-ts" + str(args.target_scale) + "-ss" + str(args.score_scale) + "-ap" + str(args.ap)
    exp_name += "-dl" + str(args.delay_len) + "-clip" + str(args.target_clip)
    exp_name += "-alpha" + str(args.alpha) + "-lr" + str(args.lr)
    if args.restore_model:
        exp_name += "-restore_model" + str(args.restore_model)

    mpi_fork(args.cpu, bind_to_core=False, cpu_set="")  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    from game_env.new_env import TradingEnv, FrameStack

    env = TradingEnv(data_v=args.data_v, action_scheme_id=args.action_scheme, obs_dim=args.obs_dim,
                     auto_follow=args.auto_follow, max_ep_len=args.max_ep_len, delay_len=args.delay_len,
                     target_scale=args.target_scale, score_scale=args.score_scale, profit_scale=args.profit_scale,
                     action_punish=args.ap, target_clip=args.target_clip, burn_in=args.burn_in)

    if args.num_stack > 1:
        env = FrameStack(env, args.num_stack, jump=5, model=args.model)

    if args.model == 'mlp':
        actor_critic = core.mlp_actor_critic
    else:
        actor_critic = core.cnn_actor_critic

    ppo(
        lambda: env,
        data_v=args.data_v,
        actor_critic=actor_critic,
        alpha=args.alpha,
        ac_kwargs=dict(hidden_sizes=args.hidden_sizes), gamma=args.gamma, lr=args.lr, ap=args.ap,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, max_ep_len=args.max_ep_len,
        logger_kwargs=logger_kwargs, exp_name=exp_name, restore_model=args.restore_model)

    os._exit(8)
