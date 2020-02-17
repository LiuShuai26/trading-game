import numpy as np
import gym
from gym import spaces
import ctypes
import json


price_mean = 2.687332e+04
price_max = 2.854000e+04
volume_mean = 5.847698e+04
volume_max = 2.052790e+05
target_mean = 2.942739e+00
target_max = 3.390000e+02
price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 30, 32, 34, 36, 38]
volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 29, 31, 33, 35, 37, 39]
target_filter = [26, 27]


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

        n_actions = 30
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(38,), dtype=np.float32)

    def reset(self, start_day=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        if not start_day:
            start_day = np.random.randint(1, 62, 1)[0]
        skip_step = 0
        start_info = {"date_index": f"{start_day} - {start_day}", "skip_steps": skip_step}
        if self.ctx:
            self.close()
        self.ctx = self.game_so.CreateContext(json.dumps(start_info).encode())
        self.game_so.GetActions(self.ctx, self.actions, self.action_len)
        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        obs = self._get_obs(self.raw_obs)
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array(obs).astype(np.float32)

    def step(self, action_index):
        if action_index < self.action_len[0]:
            self.game_so.Action(self.ctx, self.actions[action_index])
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.game_so.Step(self.ctx)

        self.game_so.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.game_so.GetReward(self.ctx, self.rewards, self.rewards_len)

        obs = self._get_obs(self.raw_obs)

        done = bool(self.raw_obs[0])
        reward = -self.rewards[0]  # 0:score, 1:profit, 2:total_profit

        # Optionally we can pass additional info, we are not using that for now
        info = {"TradingDay": self.raw_obs[25], "profit": self.rewards[1]}

        return obs, reward, done, info

    def _get_obs(self, raw_obs):
        obs = np.array(raw_obs[:40], dtype=np.float32)
        obs[price_filter] = (obs[price_filter]-price_mean) / price_max
        obs[volume_filter] = (obs[volume_filter] - volume_mean) / volume_max
        obs[target_filter] = (obs[target_filter] - target_mean) / target_max
        obs = np.delete(obs, [0, 25])
        return obs

    def render(self):
        # TODO
        pass

    def close(self):
        self.game_so.ReleaseContext(self.ctx)


env = TradingEnv()

while True:
    for i in range(1, 63):
        obs = env.reset(start_day=i)
        step = 1
        while True:
            action = env.action_space.sample()
            # if step < 100000:
            #     action = 3
            # else:
            #     action = 0
            # print("Step {}".format(step))
            # print("Action: ", action)
            obs, reward, done, info = env.step(action)
            step += 1
            # print('obs=', obs, 'reward=', reward, 'done=', done)
            # print('reward=', reward, 'profit=', info['profit'])

            if done:
                print("TradingDay", info["TradingDay"], "step:", step)
                break
