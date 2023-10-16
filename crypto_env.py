import gym
from gym.spaces import Discrete, Box
import pandas as pd
import numpy as np
import random
from pymongo import MongoClient


class CryptoEnv(gym.Env):
    def __init__(self, config=None):

        self.observation_space = Box(-np.inf, np.inf, shape=(config['NUM_STATES'],), dtype=np.float32)
        self.action_space = Discrete(config['NUM_ACTIONS'])
        self.max_fee = config['FEE']
        self.max_ep = config['MAX_EP']
        self.last_action = None

        self.num_states = config['NUM_STATES']
        self.frameskip = config['frameskip']
        self._period0=None
        self._period1=None
        try:
            self.worker_idx = config.worker_index
            self.num_workers = config.num_workers
            self.vector_env_index = config.vector_index
        except:
            self.worker_idx = 1
            self.num_workers = 1
            self.vector_env_index = 1
        self.cumsum = 0.0
        self.seed_val = (self.worker_idx - 1) + (self.num_workers * self.vector_env_index)
        self.seed(self.seed_val)
        self.mode = config['mode']
        self.client = MongoClient("mongodb://ysr1004:q5n76hrh@192.168.0.10:27017")
        # self.client = MongoClient("mongodb://localhost:27017")
        if self.mode == "train":
            self.db = self.client.Binance
            self.collection = self.db.Binance
        else:
            self.db = self.client.Binance_test
            self.collection = self.db.Binance_test
        self.columns = ['btcusdt_FUTURES_Open',
                        'btcusdt_FUTURES_High',
                        'btcusdt_FUTURES_Low',
                        'btcusdt_FUTURES_Close']
        self.df_size = config['DF_SIZE']
        self.last_state= np.zeros(len(self.columns))
        self.df = None
        self.col = 'btcusdt_FUTURES_adjusted15'
        self.last_action = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False
        self.stop_level = 0.8

    def step(self, action):
        info = {}
        if self.done:
            return np.zeros(self.num_states), 0, self.done, {}

        if isinstance(self.frameskip, int):
            self.num_steps = self.frameskip
        else:
            self.num_steps = np.random.randint(self.frameskip[0], self.frameskip[1]+1)

        self._period0 = self._period1
        self._period1 += self.num_steps

        reward = self._take_action(action)
        self.refresh_max_wallet(reward)
        obs = self.getState()

        if (self.cumsum - self.max_wallet) <= np.log(self.stop_level):
            self.done = True

        if self._period1 >= len(self.df):
            self.done = True
            info={}

        return obs, reward, self.done, info

    def refresh_max_wallet(self, reward):
        self.cumsum += reward
        if self.max_wallet < self.cumsum:
            self.max_wallet = self.cumsum


    def reset(self):

        if self.mode=="train":
            # self.start_point = 0

            # start_idx = np.random.randint(self.df_size//self.max_ep)
            # self.start_point = start_idx * self.max_ep

            self.start_point = np.random.randint(0, self.df_size - 32 - 1)
        else:
            self.start_point = np.random.randint(0, self.df_size - 32 - 1)



        self.df = pd.DataFrame(list(self.collection.find(filter={"_id":
                                                                       {"$gte":
                                                                            self.start_point}
                                                                   },
                                                         sort=[("_id", 1)],
                                                         limit=self.max_ep)
                                    )
                               )

        self.last_action = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False
        if self.max_fee != 0.0:
            self.fee = np.log(1-(np.random.normal(loc = self.max_fee, scale=self.max_fee*0.1)/100))
        else:
            self.fee = 0.0

        if isinstance(self.frameskip, int):
            self.num_steps = self.frameskip
        else:
            self.num_steps = np.random.randint(self.frameskip[0], self.frameskip[1]+1)

        self._period0 = 0
        self._period1 = self.num_steps

        state = self.getState()

        # clip_value = self.df.loc[:, 'btcusdt_FUTURES_ret'].iloc[self._period1:]
        # self.df.loc[:, 'btcusdt_FUTURES_ret'] = clip_value - np.nanmean(clip_value)

        return state

    def _take_action(self, action):
        last_signal = [0, 1, -1][self.last_action]
        signal = [0, 1, -1][action]
        signal_gap = abs(last_signal - signal)

        gap = np.nansum(self.df.iloc[self._period0:self._period1][self.col].values)
        reward = 0.0
        reward += self.fee * signal_gap
        reward += signal * gap

        self.last_action = action

        return reward
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
    def getState(self):
        raw_price = self.df.iloc[self._period0:self._period1][self.columns].values

        _o = raw_price[0,0]
        _h = np.max(raw_price[:,1])
        _l = np.min(raw_price[:,2])
        _c = raw_price[-1,-1]

        price_ohlc = np.array((_o,_h,_l,_c))
        diff_matrix = np.subtract.outer(price_ohlc, price_ohlc)
        log_price_ohlc = np.log(price_ohlc)
        log_diff_matrix = np.subtract.outer(log_price_ohlc, log_price_ohlc)
        self.state = np.zeros(self.num_states)
        # self.state[0:4] = price_ohlc
        # self.state[4:8] = log_price_ohlc
        self.state[0:6] = diff_matrix[np.triu_indices(4, k = 1)]
        self.state[6:12] = log_diff_matrix[np.triu_indices(4, k=1)]
        # self.state[:6] = diff_matrix[np.triu_indices(4, k = 1)]
        self.state[-1] = self.fee
        return self.state

    def render(self, mode='human', close=False):
        pass
