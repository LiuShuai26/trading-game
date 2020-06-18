import gym
from collections import deque
import numpy as np
from gym import spaces
from gym.spaces import Box


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, delay_len=30, target_clip=5, target_scale=1, score_scale=1.5, action_punish=0.5,
                 start_day=None, start_skip=None, duration=None, burn_in=3000):
        super(EnvWrapper, self).__init__(env)
        # target
        self.target_diff = deque(maxlen=delay_len)  # target delay setting
        self.target_clip = target_clip
        # reward
        self.target_scale = target_scale
        self.score_scale = score_scale
        self.ap = action_punish
        # env reset
        self.start_day = start_day
        self.start_skip = start_skip
        self.duration = duration
        self.burn_in = burn_in
        # statistic
        self.act_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                        15: 0, 16: 0}

    def baseline069(self, raw_obs):
        if raw_obs[26] > raw_obs[27]:
            action = 6
        elif raw_obs[26] < raw_obs[27]:
            action = 9
        else:
            action = 0
        return action

    def _env_skip(self, burn_in):
        for _ in range(burn_in):
            # action_index = self.baseline069(self.raw_obs)
            action_index = 0
            self.env.expso.Action(self.ctx, self.actions[action_index])
            self.env.expso.Step(self.ctx)
        self.env.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)

    def reset(self, start_day=None, start_skip=None, duration=None, burn_in=0):
        self.act_sta = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                        15: 0, 16: 0}
        if start_day is not None:
            obs = self.env.reset(start_day=start_day, start_skip=start_skip, duration=duration, burn_in=burn_in)
            self._env_skip(burn_in)
        else:
            obs = self.env.reset(start_day=self.start_day,
                                 start_skip=self.start_skip,
                                 duration=self.duration,
                                 burn_in=self.burn_in)
            self._env_skip(self.burn_in)
        return obs

    def step(self, action):

        last_target = self.raw_obs[26]
        last_bias = self.raw_obs[26] - self.raw_obs[27]
        last_score = self.rewards[0]

        obs, _, done, _ = self.env.step(action)

        profit = self.rewards[1]
        one_step_score = self.rewards[0] - last_score
        reward_score = one_step_score * self.score_scale

        target_now = self.raw_obs[26]
        actual_target = self.raw_obs[27]

        target_bias = target_now - actual_target

        # self.target_diff是长度为【10】的队列，存放target每次的差值。队列中的target_diff的总和就是当前总容忍度
        # 与上一步的target差值相比，同号且绝对值变小，代表target向实际target靠近，此target变化不应给惩罚延迟
        if not (last_bias * target_bias >= 0 and abs(last_bias) > abs(target_bias)):
            self.target_diff.append(abs(target_now - last_target))
        target_tolerance = sum(self.target_diff)

        reward_target_bias = abs(target_bias)
        # target delay
        reward_target_bias = max(0, reward_target_bias - target_tolerance)
        # target clip
        # target_clip = round(target_now * 0.05)
        reward_target_bias = max(0, reward_target_bias - self.target_clip)
        reward_target_bias *= self.target_scale

        action_penalization = 0 if action == 0 else 1

        designed_reward = -(reward_target_bias + self.ap * action_penalization + reward_score)

        self.act_sta[action] += 1

        info = {"TradingDay": self.raw_obs[25], "profit": profit,
                "one_step_score": one_step_score,
                "score": self.rewards[0],
                "reward_score": reward_score,
                "target_bias": abs(target_bias),
                "reward_target_bias": reward_target_bias,
                "reward_ap_num": action_penalization * self.ap,
                "target_total_tolerance": target_tolerance+self.target_clip,
                }

        return obs, designed_reward, done, info
