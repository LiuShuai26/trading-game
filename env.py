import numpy as np
import gym
from gym import spaces
import ctypes
import json
import os
from collections import deque
import pandas as pd
import time

os.chdir("/home/shuai/trading-game/rl_game/game")

info_names = [
    "Done", "LastPrice", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2",
    "AskPrice2", "AskVolume2", "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3", "BidPrice4",
    "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5", "Volume",
    "HighestPrice", "LowestPrice", "TradingDay", "Target_Num", "Actual_Num", "AliveBidPrice1",
    "AliveBidVolume1", "AliveBidPrice2", "AliveBidVolume2", "AliveBidPrice3", "AliveBidVolume3",
    "AliveAskPrice1", "AliveAskVolume1", "AliveAskPrice2", "AliveAskVolume2", "AliveAskPrice3",
    "AliveAskVolume3", "score", "profit", "total_profit", "baseline_profit", "action", "designed_reward"
]

data_len = [
    225016, 225018, 225018, 225018, 225018, 225017, 225018, 225016, 225014, 225016, 225016, 225018, 225018, 225015,
    225018, 225016, 177490, 225016, 225018, 225016, 225016, 225016, 225018, 225016, 225018, 225018, 225016, 225016,
    225016, 225018, 225018, 225016, 225016, 225018, 225016, 225016, 225018, 225016, 225016, 225015, 225016, 225016,
    225016, 225016, 192623, 225018, 225018, 225016, 225016, 225016, 225016, 225018, 225016, 225018, 225016, 225016,
    225016, 225016, 99006, 225016, 225018, 99010
]


class TradingEnv(gym.Env):

    def __init__(self, num_stack=1):
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

        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        self.n_actions = 15
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(num_stack, 38,), dtype=np.float32)

        self.max_ep_len = 3000
        self.render = False
        self.analyse = False
        self.all_data = []
        self.obs = None
        self.target_diff = deque(maxlen=30)

    def _framestack(self, observation):
        if not self.frames:
            [self.frames.append(observation) for _ in range(self.num_stack)]
        else:
            self.frames.append(observation)
        return np.stack(self.frames, axis=0)

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
            skip_step = int(np.random.randint(0, data_len[data_len_index] - (self.max_ep_len+1010), 1)[0])
        start_info = {"date_index": f"{start_day} - {start_day}", "skip_steps": skip_step}
        if self.ctx:
            self.close()
        self.ctx = self.game_so.CreateContext(json.dumps(start_info).encode())
        self.game_so.GetActions(self.ctx, self.actions, self.action_len)
        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)

        # self.target_queue.append(self.raw_obs[26])

        self.obs = self._get_obs(self.raw_obs)

        if self.analyse:
            self._append_one_step_data()
        if self.render:
            self.rendering()
        if self.num_stack > 1:
            self.obs = self._framestack(self.obs)
        return self.obs

    def _step(self, action_index):

        # 根据买卖方向进行反方向撤单操作
        if 1 <= action_index <= 7:
            self.game_so.Action(self.ctx, self.actions[18])  # 如果是买动作，卖方向全撤。
        else:
            self.game_so.Action(self.ctx, self.actions[15])  # 如果是卖动作，买方向全撤。

        self.game_so.Action(self.ctx, self.actions[action_index])

    def step(self, action_index):
        last_target = self.raw_obs[26]
        last_bias = self.raw_obs[26] - self.raw_obs[27]

        if action_index < self.n_actions:
            self._step(action_index)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action_index))

        self.game_so.Step(self.ctx)

        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.game_so.GetReward(self.ctx, self.rewards, self.rewards_len)

        self.obs = self._get_obs(self.raw_obs)
        if self.num_stack > 1:
            self.obs = self._framestack(self.obs)

        done = bool(self.raw_obs[0])

        score = self.rewards[0]
        profit = self.rewards[1]
        baseline_profit = self.rewards[3]

        # TODO 需要考虑target变化的方向
        # self.target_diff是长度为【10】的队列，存放target每次的差值。队列中的target_diff的总和就是当前总容忍度
        target_bias = self.raw_obs[26] - self.raw_obs[27]
        # 与上一步的target差值相比，同号且绝对值变小，代表target向实际target靠近，此target变化不应给惩罚延迟
        if not (last_bias * target_bias >= 0 and last_bias > target_bias):
            self.target_diff.append(abs(self.raw_obs[26] - last_target))
        target_tolerance = sum(self.target_diff)
        target_bias = abs(target_bias)
        target_bias = 0 if target_bias < target_tolerance else target_bias - target_tolerance
        action_penalization = 0 if action_index == 0 else 0.005
        # designed_reward = -score - target_bias  # score smaller better, target_bias smaller better.
        designed_reward = -(target_bias + action_penalization + score/1000)
        # Optionally we can pass additional info, we are not using that for now
        info = {"TradingDay": self.raw_obs[25], "profit": profit, "score": score, "target_bias": target_bias,
                "ap_num": action_penalization/0.005}

        if self.analyse:
            self._append_one_step_data(action=action_index, designed_reward=designed_reward)
        if self.render:
            self.rendering(action_index)

        return self.obs, designed_reward, done, info

    def _get_obs(self, raw_obs):
        price_mean = 26871.05
        price_max = 28540.0
        bid_ask_volume_log_mean = 2.05
        bid_ask_volume_log_max = 6.43
        total_volume_mean = 56871.13
        total_volume_max = 175383.0
        target_mean = 20.69
        target_max = 485.0
        price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 30, 32, 34, 36, 38]
        bid_ask_volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 31, 33, 35, 37, 39]
        total_volume_filter = [22]
        target_filter = [26, 27]
        obs = np.array(raw_obs[:44], dtype=np.float32)

        obs = np.delete(obs, [34, 35, 42, 43])

        obs[price_filter] = (obs[price_filter] - price_mean) / (price_max - price_mean)
        obs[bid_ask_volume_filter] = (np.log(obs[bid_ask_volume_filter]) - bid_ask_volume_log_mean) / (
                bid_ask_volume_log_max - bid_ask_volume_log_mean)
        obs[total_volume_filter] = (obs[total_volume_filter] - total_volume_mean) / (
                total_volume_max - total_volume_mean)
        obs[target_filter] = (obs[target_filter] - target_mean) / (target_max - target_mean)

        obs = np.delete(obs, [0, 25])
        obs[obs < -1] = -1
        obs[obs > 1] = 1

        return obs

    def _append_one_step_data(self, action=None, designed_reward=None):
        info_dict = {}
        raw_obs = np.array(self.raw_obs[:44], dtype=np.float32)
        raw_obs = np.delete(raw_obs, [34, 35, 42, 43])
        for i in range(40):
            info_dict[info_names[i]] = raw_obs[i]

        obs_names = info_names.copy()
        del obs_names[25]
        del obs_names[0]
        for i in range(38):
            info_dict[obs_names[i] + "_n"] = self.obs[i]
        for i in range(4):
            info_dict[info_names[i + 40]] = self.rewards[i]
        info_dict[info_names[44]] = action
        info_dict[info_names[45]] = designed_reward
        self.all_data.append(info_dict)

    def rendering(self, action=None):
        print("-----------------------")
        print("Action:", action)
        print("AliveAskPriceNUM:", self.raw_obs[42])
        print("AliveAskVolumeNUM:", self.raw_obs[43])
        print("AliveAskPrice3:", self.raw_obs[40])
        print("AliveAskVolume3:", self.raw_obs[41])
        print("AliveAskPrice2:", self.raw_obs[38])
        print("AliveAskVolume2:", self.raw_obs[39])
        print("AliveAskPrice1:", self.raw_obs[36])
        print("AliveAskVolume1:", self.raw_obs[37])
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
        print("AliveBidPriceNUM:", self.raw_obs[34])
        print("AliveBidVolumeNUM:", self.raw_obs[35])
        print("-----------------------")

    def close(self):
        self.game_so.ReleaseContext(self.ctx)


if __name__ == "__main__":

    env = TradingEnv()

    cnt = 0

    while True:
        # for i in range(1, 63):
        while True:
            cnt += 1
            # obs = env.reset(render=True, analyse=True)
            obs = env.reset()
            print(env.raw_obs[26], env.raw_obs[27])
            step = 1
            while True:
                # action = env.action_space.sample()
                action = 0
                obs, reward, done, info = env.step(action)
                step += 1
                print(env.raw_obs[26], env.raw_obs[27], reward)
                if done or step == 1000:
                    print("Done!", cnt)
                    # all_data = env.all_data
                    # all_data_df = pd.DataFrame(all_data)
                    # print(all_data_df.tail())
                    break
