import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import pandas as pd
import numpy as np
import random
from pymongo import MongoClient

def truncated_normal():
    random_number = np.random.normal()
    if abs(random_number) < 2.0:
        return random_number
    else:
        return truncated_normal()

class CryptoEnv(gym.Env):
    def __init__(self, config=None):

        self.observation_space = Box(-np.inf, np.inf, shape=(config['NUM_STATES'],), dtype=np.float32)
        self.action_space = Discrete(config['NUM_ACTIONS'])
        self.max_fee = config['FEE']
        self.max_ep = config['MAX_EP']


        self.num_states = config['NUM_STATES']
        self.frameskip = config['frameskip']
        self.risk_free = np.log(1 - 0.05) / 250 / 24 / 60 * self.frameskip
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
                        'btcusdt_FUTURES_Close',]
        self.df_size = config['DF_SIZE']
        self.last_state= np.zeros(len(self.columns))
        self.df = None
        self.col = 'btcusdt_FUTURES_adjusted'
        self.last_signal = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False
        self.stop_level = 0.5
        self.fee_count = 0

    def step(self, action):
        info = {}
        if self.done:
            return np.zeros(self.num_states), 0, self.done, {}

        self.get_step()
        self._period0 = self._period1
        self._period1 += self.num_steps

        reward = self._take_action(action)
        self.refresh_max_wallet(reward)
        obs = self.getState()
        truncated = False

        if (self.cumsum - self.max_wallet) <= np.log(self.stop_level):
            self.done = True

        if self._period1 >= len(self.df):
            self.done = True
            truncated = True
            info={}

        if self.done and (self.fee_count <= 100):
            print(self.fee_count, self.fee)

        return (obs, reward, self.done, truncated, info)

    def refresh_max_wallet(self, reward):
        self.cumsum += reward
        if self.max_wallet < self.cumsum:
            self.max_wallet = self.cumsum

    def get_step(self):
        self.num_steps = random.randint(self.frameskip-1, self.frameskip+1)
        # self.num_steps = self.frameskip
    def reset(self, *, seed=None, options=None):
        self.fee_count+=1

        if self.mode=="train":
            self.start_point = np.random.randint(0, self.df_size - 32 - 1)
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
        count_max = 100
        if self.max_fee != 0.0:
            fee_loc = min(count_max, self.fee_count)/count_max*self.max_fee
            fee_scale = fee_loc/2

            self.fee = fee_loc + truncated_normal() * fee_scale
            self.logfee = np.log(1-(self.fee/100))
        else:
            self.fee = 0.0
            self.logfee = 0.0

        state = self.getState()

        clip_value = self.df.loc[:, self.col].iloc[self._period1:]
        self.df.loc[:, self.col] = clip_value - np.nanmean(clip_value)

        return state, {}

    def _take_action(self, action):
        #action = 0,1,2

        signal = [0, 1, -1][action]
        signal_gap = abs(self.last_signal - signal)
        _ = self.df.iloc[self._period0:self._period1][self.col].values
        gap = np.nansum(self.df.iloc[self._period0:self._period1][self.col].values)
        reward = 0.0
        reward += self.logfee * signal_gap
        reward += signal * gap
        # reward += [1,0,0][action] * self.risk_free

        self.last_signal = signal

        return reward

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
    def getState(self):
        raw_price = self.df.iloc[self._period0:self._period1][self.columns].values

        _o = raw_price[0,0]
        _h = np.max(raw_price)
        _l = np.min(raw_price)
        _c = raw_price[-1,-1]
        price_ohlc = np.array((_o, _h, _l, _c))
        log_price_ohlc = np.log(price_ohlc)

        log_diff_matrix = np.subtract.outer(log_price_ohlc, log_price_ohlc)
        self.state = np.zeros(self.num_states)
        # self.state[0] = (log_price_ohlc[-1,-1] - log_price_ohlc[0,0]) # close - open
        # self.state[1] = np.max(log_price_ohlc[:,1]) - np.min(log_price_ohlc[:,2]) # high - low
        # self.state[2] = np.max(log_price_ohlc[:, 1]) - log_price_ohlc[-1,-1]  # high - close
        # self.state[0:4] = price_ohlc
        # self.state[4:8] = log_price_ohlc
        self.state[0:6] = log_diff_matrix[np.triu_indices(4, k = 1)]
        # self.state[0:36] = log_diff_matrix[np.triu_indices(9, k=1)]
        # self.state[:6] = diff_matrix[np.triu_indices(4, k = 1)]

        self.state[-1] = self.logfee
        return self.state

    def render(self, mode='human', close=False):
        pass
