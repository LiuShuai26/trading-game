import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
pd.set_option('display.max_columns', None)
from collections import deque
import numpy as np
import time
import os.path as osp
import tensorflow as tf
import sys
import os
ROOT = sys.path[0]
sys.path.append(ROOT+'/spinningup')
from spinup.utils.logx import restore_tf_graph
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""


class DataV:
    def __init__(self, mode="game", csv_path="/home/shuai/day1-baseline_policy_au.csv"):
        # self.step = deque(maxlen=100)
        # self.last_price = deque(maxlen=100)
        # self.target = deque(maxlen=100)
        # self.actual_target = deque(maxlen=100)
        # self.action = deque(maxlen=100)
        # self.bid = deque(maxlen=100)
        # self.ask = deque(maxlen=100)
        # self.target_punish = deque(maxlen=100)
        self.step = []
        self.last_price = []
        self.target = []
        self.actual_target = []
        self.action = []
        self.bid = []
        self.ask = []
        self.target_punish = []
        self.ask_price = []
        self.bid_price = []
        self.action_price = [999, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3]

        self.mode = mode
        assert mode in ['game', 'csv'], "mode must be game or csv"
        if mode is 'game':
            self.env = self.make_env(num_stack=1, obs_dim=26, actions=15)
            self.get_action = self.load_model()
            self.o = None
            print("successfully make env and load model!")
        else:
            self.csv_data = pd.read_csv(csv_path)
            print("successfully load csv!")

        # Create figure for plotting
        self.fig, self.axes = plt.subplots(2, 2)

    def make_env(self, num_stack, obs_dim, actions):
        from trading_env import TradingEnv, FrameStack
        from wrapper import EnvWrapper
        env = TradingEnv(action_scheme_id=actions, obs_dim=obs_dim)
        if num_stack > 1:
            env = FrameStack(env, num_stack)
        env = EnvWrapper(env)
        return env

    def load_model(self):
        fpath = ROOT + "/spinningup/data/"
        model = 'tf1_save190.080'
        fname = osp.join(fpath, model)

        print('\n\nLoading from %s.\n\n ' % fname)
        # load the things!
        sess = tf.Session()
        model = restore_tf_graph(sess, fname)
        print('Using default action op.')
        action_op = model['pi']

        # make function for producing an action given a single state
        get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]
        return get_action

    # This function is called periodically from FuncAnimation
    def animate(self, i):

        if self.mode is 'game':
            self.append_game_data(i)
        else:
            self.append_csv_data(i)

        self.axes[0][0].clear()
        # last price
        self.axes[0][0].plot(self.step[-50:], self.last_price[-50:])
        # action in last price

        self.axes[0][0].scatter(self.step[-50:], self.ask_price[-50:], c=self.ask[-50:], cmap="Greens", s=6)
        self.axes[0][0].scatter(self.step[-50:], self.bid_price[-50:], c=self.bid[-50:], cmap="Reds", s=3)

        self.axes[1][0].clear()
        self.axes[1][0].plot(self.step[-50:], self.target[-50:])
        self.axes[1][0].plot(self.step[-50:], self.actual_target[-50:])
        self.axes[1][0].scatter(self.step[-50:], np.array(self.target[-50:]) + np.array(self.target_punish[-50:]), marker="s", c='red', s=10)
        self.axes[1][0].scatter(self.step[-50:], np.array(self.target[-50:]) - np.array(self.target_punish[-50:]), marker="s", c='red', s=10)

        self.axes[0][1].clear()
        self.axes[0][1].scatter(self.step[-50:], self.action[-50:], c=self.ask[-50:], cmap="Greens", s=60)
        self.axes[0][1].scatter(self.step[-50:], self.action[-50:], c=self.bid[-50:], cmap="Reds", s=30)

        # self.axes[1][1].clear()
        # self.axes[1][1].scatter(self.step[-50:], self.delay_action[-50:], cmap="Greens", s=60)
        # self.axes[1][1].scatter(self.step[-50:], self.delay_action[-50:], cmap="Reds", s=30)

    def append_game_data(self, i):
        if i == 0:
            self.o = self.env.reset(start_day=10, start_skip=0)

        delay_a = 0
        if abs(self.env.raw_obs[26] - self.env.raw_obs[27]) > 5:
            if self.env.raw_obs[26] > self.env.raw_obs[27]:
                delay_a = 6
            else:
                delay_a = 9

        a = self.get_action(self.o)
        self.o, r, d, info = self.env.step(a)

        self.step.append(i)
        self.last_price.append(self.env.raw_obs[1])
        self.target.append(self.env.raw_obs[26])
        self.actual_target.append(self.env.raw_obs[27])
        self.action.append(a)
        self.target_punish.append(info['target_total_tolerance'])

        if 1 <= a <= 7:
            self.bid.append(8 - a + 10)   # 11-17
            self.bid_price.append(self.action_price[a] + self.env.raw_obs[1])
            self.ask.append(0)
            self.ask_price.append(self.env.raw_obs[1])
        elif 8 <= a <= 14:
            self.ask.append(a - 7 + 10)   # 11-17
            self.ask_price.append(self.action_price[a] + self.env.raw_obs[1])
            self.bid.append(0)
            self.bid_price.append(self.env.raw_obs[1])
        else:
            self.ask.append(0)
            self.ask_price.append(self.env.raw_obs[1])
            self.bid_price.append(self.env.raw_obs[1])
            self.bid.append(0)

    def append_csv_data(self, i):
        self.step.append(i)
        self.last_price.append(self.csv_data['LastPrice'].iloc[i])
        self.target.append(self.csv_data['Target_Num'].iloc[i])
        self.actual_target.append(self.csv_data['Actual_Num'].iloc[i])
        a = self.csv_data['action'].iloc[i]
        self.action.append(a)
        if 1 <= a <= 7:
            self.bid.append(8 - a + 10)
            self.ask.append(0)
        elif 8 <= a <= 14:
            self.ask.append(a - 7 + 10)
            self.bid.append(0)
        else:
            self.ask.append(0)
            self.bid.append(0)

    def animation_show(self):

        ani_running = True

        def onClick(event):
            nonlocal ani_running
            if ani_running:
                ani.event_source.stop()
                ani_running = False
            else:
                ani.event_source.start()
                ani_running = True

        self.fig.canvas.mpl_connect('button_press_event', onClick)

        # Set up plot to call animate() function periodically
        ani = animation.FuncAnimation(self.fig, self.animate, interval=1)
        ani.event_source.stop()
        plt.show()

    def show(self):
        for i in range(1000):
            self.append_game_data(i)
        self.axes[0][0].plot(self.step, self.last_price)
        self.axes[0][0].scatter(self.step, self.last_price, c=self.ask, cmap="Greens", s=60)
        self.axes[0][0].scatter(self.step, self.last_price, c=self.bid, cmap="Reds", s=30)

        self.axes[1][0].plot(self.step, self.target)
        self.axes[1][0].plot(self.step, self.actual_target)
        self.axes[1][0].scatter(self.step,
                                np.array(self.target) + np.array(self.target_punish), marker="s",
                                c='red', s=10)
        self.axes[1][0].scatter(self.step,
                                np.array(self.target) - np.array(self.target_punish), marker="s",
                                c='red', s=10)

        self.axes[0][1].scatter(self.step, self.action, c=self.ask, cmap="Greens", s=60)
        self.axes[0][1].scatter(self.step, self.action, c=self.bid, cmap="Reds", s=30)
        plt.show()


if __name__ == '__main__':
    mode = 'game'   # 'game' or 'csv'
    animate = True

    testv = DataV(mode=mode)
    if animate:
        testv.animation_show()
    else:
        testv.show()
    os._exit(8)
