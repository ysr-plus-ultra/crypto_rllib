import time

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium.spaces import Discrete, Box, Dict
from pymongo import MongoClient

load_dotenv()
import os


class CryptoEnv(gym.Env):
    def __init__(self, config=None, worker_idx=0, manual=False):
        super(CryptoEnv, self).__init__()
        self.config = config
        self.truncated = None
        self.done = None
        self.start_point = None
        self.df = None
        self.step_idx = None
        self.state_stack = None
        self.stop_level = 0.8
        self.last_reward = None
        self.last_action = None
        self.start_time = time.time()

        self._setup_spaces()
        self.mode = self.config.get('MODE', "train")
        self.max_fee = self.config.get('FEE', 0.0) / 100
        self.max_ep = self.config.get('MAX_EP', 1000)
        if self.mode == "train":
            self.df_size = self.config.get('DF1_SIZE', 1000)
        elif self.mode == "eval":
            self.df_size = self.config.get('DF2_SIZE', 1000)

        self.frame = self.config.get('FRAME', 1)
        self.timeframe_adjust = np.sqrt(43200)

        self.worker_idx = getattr(config, 'worker_index', worker_idx)
        self.num_workers = getattr(config, 'num_workers', 0)
        self.vector_env_index = getattr(config, 'vector_index', 0)

        self.seed_val =  self.worker_idx + (self.num_workers * self.vector_env_index)

        super().reset(seed=self.seed_val)

        self.cumsum = 0.0
        self.first_run = self._np_random.integers(0, self.max_ep+1).item()
        print(self.worker_idx + (self.num_workers * self.vector_env_index), self.first_run)

        if not manual:
            self.client = MongoClient(os.getenv('MONGO_URL'))

            if self.mode == "train":
                self.db = self.client.Binance
                self.collection = self.db.Binance
            elif self.mode == "eval":
                self.db = self.client.Binance_valid
                self.collection = self.db.Binance_valid
        self.columns = ['btcusdt_FUTURES_Open',
                        'btcusdt_FUTURES_High',
                        'btcusdt_FUTURES_Low',
                        'btcusdt_FUTURES_Close']

        self.col = 'btcusdt_FUTURES_ret'

    def _setup_spaces(self):
        """Sets up action and observation spaces."""
        self.action_space = Discrete(3)

        self.observation_space = Dict({
            '0_price': Dict({
                "0_open": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "1_high": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "2_low": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "3_close": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }),
            '1_last_action': self.action_space,
            '2_last_reward': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            '3_fee': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    def reset(self, *, seed=None, options=None, target_df=None):
        super().reset(seed=seed, options=options)
        self.reward_sum = 0.0
        if self.mode == "train":
            self.start_point = self._np_random.integers(0, self.df_size - self.max_ep + 1).item()
        else:
            self.start_point = 0
        if target_df is None:
            self.df = pd.DataFrame(list(self.collection.find(filter={"_id":
                                                                         {"$gte":
                                                                              self.start_point}
                                                                     },
                                                             sort=[("_id", 1)],
                                                             limit=self.max_ep)
                                        )
                                   )
        else:
            self.df = target_df
        self.done = False
        self.truncated = False
        self.cumsum = 0.0
        self.max_wallet = 0.0
        self.step_idx = 0

        self.get_step()

        self._period0 = 0
        self._period1 = self.num_steps

        self.gap_stack = self.df[self.col].to_numpy(copy=True)
        self.gap_stack[:self._period1] = np.nan
        self.state_stack = self.df[self.columns].to_numpy(copy=True)
        self.gap_mu = 0.0

        # detrending

        clip_value = self.gap_stack[self._period1:]
        self.gap_mu = np.nanmean(clip_value)
        self.gap_stack -= self.gap_mu

        self.set_fee()
        self.last_price = None
        self.last_action = 0
        self.last_signal = 0
        self.last_reward = 0.0
        price = self.getPrice()

        new_state = {
            '0_price': {
                '0_open': np.array(price[0], dtype=np.float32).reshape(-1),
                '1_high': np.array(price[1], dtype=np.float32).reshape(-1),
                '2_low': np.array(price[2], dtype=np.float32).reshape(-1),
                '3_close': np.array(price[3], dtype=np.float32).reshape(-1),
            },
            '1_last_action': 0,
            '2_last_reward': np.array(0.0, dtype=np.float32).reshape(-1),
            '3_fee': np.array(self.logfee * np.sqrt(43200), dtype=np.float32).reshape(-1),
        }

        return new_state, {}

    def step(self, action):
        info = {}

        self.get_step()
        self._period0 = self._period1
        self._period1 += self.num_steps

        raw_reward, normal_reward, raw_reward_without_fee = self._take_action(action)

        self.last_action = action
        self.last_reward = raw_reward_without_fee

        price = self.getPrice()
        self.set_fee()
        new_state = {
            '0_price': {
                '0_open': np.array(price[0], dtype=np.float32).reshape(-1),
                '1_high': np.array(price[1], dtype=np.float32).reshape(-1),
                '2_low': np.array(price[2], dtype=np.float32).reshape(-1),
                '3_close': np.array(price[3], dtype=np.float32).reshape(-1),
            },
            '1_last_action': self.last_action,
            '2_last_reward': np.array(self.last_reward, dtype=np.float32).reshape(-1),
            '3_fee': np.array(self.logfee * np.sqrt(43200), dtype=np.float32).reshape(-1),
        }

        # drawdown
        if (self.cumsum - self.max_wallet) <= np.log(self.stop_level):
            self.done = True

        return new_state, normal_reward, self.done, self.truncated, info

    def refresh_max_wallet(self, reward):
        self.cumsum += reward
        if self.max_wallet < self.cumsum:
            self.max_wallet = self.cumsum

    def set_fee(self):
        if self.mode == "train":
            target_loc = min(max((time.time() - self.start_time)/3600 - 1, 0.0), 3.0) / 3.0 * self.max_fee
            self.fee = self._np_random.normal(loc=target_loc, scale=self.max_fee/2)
        else:
            self.fee = self.max_fee
        self.logfee = np.log1p(-self.fee)

    def get_step(self):
        self.num_steps = self.frame

    def _take_action(self, action):
        # action = 0,1,2

        signal = [0, 1, -1][action]
        signal_gap = abs(self.last_signal - signal)
        _ = self.gap_stack[self._period0:self._period1]
        raw_gap = np.nansum(_ + self.gap_mu)
        normal_gap = np.nansum(_)

        raw_reward = signal * raw_gap + self.logfee * signal_gap
        normal_reward = signal * normal_gap + self.logfee * signal_gap
        raw_reward_without_fee = signal * raw_gap

        self.last_signal = signal
        self.refresh_max_wallet(raw_reward)

        return (raw_reward * self.timeframe_adjust,
                normal_reward * self.timeframe_adjust,
                raw_reward_without_fee * self.timeframe_adjust)

    def getPrice(self):
        state_stack = self.state_stack[self._period0:self._period1]
        if self._period1 == len(self.state_stack):
            self.truncated = True

        if self.mode == "train" and self.first_run is not None:
            if self._period1 >= self.first_run:
                self.first_run = None
                self.truncated = True

        price_stack = state_stack.flatten()

        _h_idx = np.argmax(price_stack)
        _l_idx = np.argmin(price_stack)

        ohlc = np.array((price_stack[0], price_stack[_h_idx], price_stack[_l_idx], price_stack[-1]))

        if self.last_price is None:
            self.last_price = ohlc

        div_ohlc = np.log(ohlc / self.last_price) * self.timeframe_adjust
        self.last_price = ohlc

        return div_ohlc

    def render(self, mode='human', close=False):
        pass
