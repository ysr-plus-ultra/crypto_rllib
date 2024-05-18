import random

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete, Box
from pymongo import MongoClient
import itertools
import nb.auth_file as myauth

class CryptoPortfolioEnv(gym.Env):
    symbols = ['btcusdt', 'ethusdt', 'adausdt', 'solusdt', 'bnbusdt']
    col = ["_FUTURES_Open", "_FUTURES_High", "_FUTURES_Low", "_FUTURES_Close", "_FUTURES_Volume", "_FUTURES_ret"]
    columns = ["".join(x) for x in itertools.product(symbols, col)]
    def __init__(self, config=None, worker_idx=0):

        self.observation_space = Box(-np.inf, np.inf, shape=(config['NUM_STATES'],), dtype=np.float32)
        self.action_space = Box(0, 1.0, shape=(config['NUM_ACTIONS'],), dtype=np.float32)
        self.max_fee = config['FEE']
        self.max_ep = config['MAX_EP']

        self.num_states = config['NUM_STATES']
        self.frameskip = config['frameskip']
        self.timeframe_adjust = np.sqrt(43200 / self.frameskip)
        self._period0 = None
        self._period1 = None

        try:
            self.worker_idx = config.worker_index
            self.num_workers = config.num_workers
            self.vector_env_index = config.vector_index
        except:
            self.worker_idx = worker_idx
            self.num_workers = 0
            self.vector_env_index = 0

        self.seed_val = self.worker_idx + (self.num_workers * self.vector_env_index)
        self.cumsum = 0.0

        self.mode = config['mode']
        self.client = MongoClient(myauth.mongo_url)

        if self.mode == "train":
            self.db = self.client.Binance
            self.collection = self.db.Binance
        else:
            self.db = self.client.Binance_valid
            self.collection = self.db.Binance_valid

        self.df_size = config['DF_SIZE']
        self.last_state = np.zeros(len(self.columns))
        self.df = None
        self.col = 'btcusdt_FUTURES_ret'
        self.last_signal = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False
        self.stop_level = 0.5
        self.last_price = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed_val)

        if self.mode == "train":
            self.start_point = np.random.randint(0, self.df_size - self.max_ep + 1)
            self.collection = random.choice([self.db.Binance,
                                             self.db.Binance_reverse,
                                             self.db.Binance_negative,
                                             self.db.Binance_reverse_negative])
        else:
            self.start_point = 0

        self.df = pd.DataFrame(list(self.collection.find(filter={"_id":
                                                                     {"$gte":
                                                                          self.start_point}
                                                                 },
                                                         sort=[("_id", 1)],
                                                         limit=self.max_ep)
                                    )
                               )

        self.last_signal = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False

        self.get_step()

        self._period0 = 0
        self._period1 = self.num_steps

        self.gap_stack = self.df[self.col].to_numpy(copy=True)
        self.gap_stack[:self._period1] = np.nan
        self.state_stack = self.df[self.columns].to_numpy(copy=True)

        # detrending
        # if not self.mode == "train":
        #     clip_value = self.gap_stack[self._period1:]
        #     self.gap_stack -= np.nanmean(clip_value)

        clip_value = self.gap_stack[self._period1:]
        self.gap_stack -= np.nanmean(clip_value)

        self.set_fee()
        self.last_price = None
        state = self.getState()

        return state, {}

    def step(self, action):
        info = {}
        if self.done:
            return np.zeros(self.num_states), 0, self.done, {}

        self.get_step()
        self._period0 = self._period1
        self._period1 += self.num_steps

        reward = self._take_action(action)
        self.set_fee()
        obs = self.getState()
        truncated = False

        # drawdown
        if (self.cumsum - self.max_wallet) <= np.log(self.stop_level):
            self.done = True

        if self._period1 >= len(self.df):
            self.done = True
            truncated = True
            info = {}

        return (obs, reward, self.done, truncated, info)

    def refresh_max_wallet(self, reward):
        self.cumsum += reward
        if self.max_wallet < self.cumsum:
            self.max_wallet = self.cumsum

    def set_fee(self):
        self.fee = self.max_fee
        self.logfee = np.log(1 - (self.fee / 100))
    def get_step(self):
        self.num_steps = self.frameskip

    def _take_action(self, action):
        # action = 0,1,2

        signal = [0, 1, -1][action]
        signal_gap = abs(self.last_signal - signal)
        _ = self.gap_stack[self._period0:self._period1]
        gap = np.nansum(_)
        reward = 0.0
        reward += self.logfee * signal_gap
        reward += signal * gap

        self.last_signal = signal
        self.refresh_max_wallet(reward)

        log_reward = reward
        return log_reward * self.timeframe_adjust

    def getState(self):
        state_stack = self.state_stack[self._period0:self._period1]
        price_stack = state_stack.flatten()

        if self._period1 >= len(self.df):
            self.num_steps = len(self.df) - self._period0

        _h_idx = np.argmax(price_stack)
        _l_idx = np.argmin(price_stack)

        ohlc = np.array((price_stack[0], price_stack[_h_idx], price_stack[_l_idx], price_stack[-1]))

        if self.last_price is None:
            self.last_price = ohlc

        div_ohlcv = ohlc / self.last_price
        lower_bound = -5.0 * 0.002333 * np.sqrt(self.frameskip)
        upper_bound = 5.0 * 0.002333 * np.sqrt(self.frameskip)
        x = np.clip(np.log(div_ohlcv), lower_bound, upper_bound) * self.timeframe_adjust

        self.state = x.astype("float32")
        self.last_price = ohlc
        return self.state

    def render(self, mode='human', close=False):
        pass
