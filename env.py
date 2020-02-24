import numpy as np
import gym
from gym import spaces
import ctypes
import json
import os
import pandas as pd
import time

os.chdir("/home/shuai/trading-game/rl_game/game")

info_names = ["Done", "LastPrice", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2",
              "AskPrice2", "AskVolume2", "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3", "BidPrice4",
              "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5", "Volume",
              "HighestPrice", "LowestPrice", "TradingDay", "Target_Num", "Actual_Num", "AliveBidPrice1",
              "AliveBidVolume1", "AliveBidPrice2", "AliveBidVolume2", "AliveBidPrice3", "AliveBidVolume3",
              "AliveAskPrice1", "AliveAskVolume1", "AliveAskPrice2", "AliveAskVolume2", "AliveAskPrice3",
              "AliveAskVolume3", "score ", "profit", "total_profit", "action", "designed_reward"]

data_len = [225016, 225018, 88391, 504024, 225018, 225017, 225018, 225016, 22501, 225016, 225016, 225016, 225018,
            225015, 225018, 16379, 177490, 225016, 225018, 225016, 225016, 225016, 225018,
            225016, 225018, 372414, 225016, 225016, 225016, 225018, 225018, 225016, 225016, 265205, 225016, 225016,
            225018, 225016, 225016, 225015, 225016, 225016, 225016, 225016, 192623, 225018,
            247995, 225016, 225016, 225016, 225016, 225018, 99198, 225018, 225016, 225016, 225016, 225016, 99006,
            225016, 225018, 99010, ]


class TradingEnv(gym.Env):

    def __init__(self):
        super(TradingEnv, self).__init__()

        so_file = "./game.so"
        self.game_so = ctypes.cdll.LoadLibrary(so_file)
        arr_len = 100
        arr1 = ctypes.c_int * arr_len
        arr = ctypes.c_int * 1

        self.ctx = None

        self.actions = arr1()
        self.action_len = arr()
        self.raw_obs = arr1()
        self.raw_obs_len = arr()
        self.rewards = arr1()
        self.rewards_len = arr()

        self.n_actions = 15
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(38,), dtype=np.float32)
        self.max_ep_len = 1000
        self.render = False
        self.analyse = False
        self.all_data = []

    def reset(self, start_day=None, skip_step=None, render=False, analyse=False):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        # np.random.seed(seed)
        self.render = render
        self.analyse = analyse
        self.all_data = []
        if not start_day:
            start_day = np.random.randint(1, 63, 1)[0]
        if skip_step is None:
            data_len_index = start_day - 1
            skip_step = int(np.random.randint(0, data_len[data_len_index] - self.max_ep_len, 1)[0])
        start_info = {"date_index": f"{start_day} - {start_day}", "skip_steps": skip_step}
        if self.ctx:
            self.close()
        self.ctx = self.game_so.CreateContext(json.dumps(start_info).encode())
        self.game_so.GetActions(self.ctx, self.actions, self.action_len)
        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)

        obs = self._get_obs(self.raw_obs)

        if self.analyse:
            self._append_one_step_data()
        if self.render:
            self.rendering()
        # here obs should be a numpy array float32 to make it more general (in case we want to use continuous actions)
        return obs

    def step(self, action_index):
        if action_index < self.n_actions:
            self.game_so.Action(self.ctx, self.actions[action_index])
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action_index))

        self.game_so.Step(self.ctx)

        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.game_so.GetReward(self.ctx, self.rewards, self.rewards_len)

        obs = self._get_obs(self.raw_obs)

        done = bool(self.raw_obs[0])

        score = self.rewards[0]
        profit = self.rewards[1]

        target_bias = abs(self.raw_obs[26] - self.raw_obs[27])
        # designed_reward = -score - target_bias  # score smaller better, target_bias smaller better.
        designed_reward = -target_bias
        # Optionally we can pass additional info, we are not using that for now
        info = {"TradingDay": self.raw_obs[25], "profit": profit}

        if self.analyse:
            self._append_one_step_data(action=action_index, designed_reward=designed_reward)
        if self.render:
            self.rendering(action_index)

        return obs, designed_reward, done, info

    def _get_obs(self, raw_obs):
        price_mean = 26867.75
        price_max = 28540.0
        bid_ask_volume_mean = 8.679
        bid_ask_volume_max = 620.0
        total_volume_mean = 58476.98
        total_volume_max = 205279.0
        target_mean = 2.94
        target_max = 339.0
        price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 30, 32, 34, 36, 38]
        bid_ask_volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 31, 33, 35, 37, 39]
        total_volume_filter = [22]
        target_filter = [26, 27]
        obs = np.array(raw_obs[:40], dtype=np.float32)

        obs[price_filter] = (obs[price_filter] - price_mean) / (price_max - price_mean)
        obs[bid_ask_volume_filter] = (obs[bid_ask_volume_filter] - bid_ask_volume_mean) / (
                bid_ask_volume_max - bid_ask_volume_mean)
        obs[total_volume_filter] = (obs[total_volume_filter] - total_volume_mean) / (
                total_volume_max - total_volume_mean)
        # obs[target_filter] = (obs[target_filter] - target_mean) / (target_max - target_mean)
        obs = np.delete(obs, [0, 25])
        obs[obs < -1] = -1
        obs[obs > 1] = 1

        return obs

    def _append_one_step_data(self, action=None, designed_reward=None):
        info_dict = {}
        for i in range(40):
            info_dict[info_names[i]] = self.raw_obs[i]
        for i in range(3):
            info_dict[info_names[i + 40]] = self.rewards[i]
        info_dict[info_names[43]] = action
        info_dict[info_names[44]] = designed_reward
        self.all_data.append(info_dict)

    def rendering(self, action=None):
        print("-----------------------")
        print("Action:", action)
        print("AliveAskPrice3:", self.raw_obs[38])
        print("AliveAskVolume3:", self.raw_obs[39])
        print("AliveAskPrice2:", self.raw_obs[36])
        print("AliveAskVolume2:", self.raw_obs[37])
        print("AliveAskPrice1:", self.raw_obs[34])
        print("AliveAskVolume1:", self.raw_obs[35])
        print("AskPrice1:", self.raw_obs[4])
        print("AskVolume1:", self.raw_obs[5])
        print(".....")
        print("LastPrice:", self.raw_obs[1])
        print("Actual_Num:", self.raw_obs[27])
        print(".....")
        print("BidPrice1:", self.raw_obs[2])
        print("BidVolume1:", self.raw_obs[3])
        print("AliveBidPrice1:", self.raw_obs[28])
        print("AliveBidVolume1:", self.raw_obs[29])
        print("AliveBidPrice2:", self.raw_obs[30])
        print("AliveBidVolume2:", self.raw_obs[31])
        print("AliveBidPrice3:", self.raw_obs[32])
        print("AliveBidVolume3:", self.raw_obs[33])
        print("-----------------------")

    def close(self):
        self.game_so.ReleaseContext(self.ctx)

env = TradingEnv()

cnt = 0

while True:
    # for i in range(1, 63):
    while True:
        cnt += 1
        # obs = env.reset(render=True, analyse=True)
        obs = env.reset(analyse=True)
        step = 1
        while True:
            action = env.action_space.sample()
            # action = 0
            # if step < 100000:
            #     action = 3
            # else:
            #     action = 0
            # print("Step {}".format(step))
            # print("Action: ", action)
            # print(step, "=======")
            obs, reward, done, info = env.step(action)
            # print(obs)
            # print('profit=', info['profit'], 'total_profit=', info['total_profit'])
            step += 1
            # time.sleep(1)
            # print('obs=', obs, 'reward=', reward, 'done=', done)
            # print('reward=', reward, 'profit=', info['profit'])

            if done or step == 100:
                print("Done!", cnt)
                all_data = env.all_data
                all_data_df = pd.DataFrame(all_data)
                print(all_data_df.tail())
                break
