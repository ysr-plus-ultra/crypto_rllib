import gym
from gym.spaces import Discrete, Box
import pandas as pd
import numpy as np
import random
from pymongo import MongoClient


class CryptoEnv(gym.Env):
    def __init__(self, config=None, ):

        self.ep = 0
        self.ep_mean= 1024
        self.fee = np.log(1 - (config['FEE'] / 100))
        self.lever = config['LEVERAGE']
        self.max_ep = config['MAX_EP']
        self.action_space = Discrete(3)
        self.observation_space = Box(-np.inf, np.inf, shape=(config['NUM_STATES'],), dtype=np.float32)
        self.num_states = config['NUM_STATES']
        self.frameskip = config['frameskip']
        self.num_steps = 1
        self.cumsum = 0.0
        self.mode = config['mode']
        self.df_size = config['df_size']
        self.client = MongoClient("mongodb://admin:dbstnfh123@192.168.0.201:27017")
        if self.mode == "train":
            self.db = self.client.Binance
            self.collection = self.db.Binance
        else:
            self.db = self.client.Binance_test
            self.collection = self.db.Binance_test
        self.columns = ['btc_FUTURES_Open', 'btc_FUTURES_High', 'btc_FUTURES_Low', 'btc_FUTURES_Close']
        self.last_state= np.zeros(len(self.columns))
        self.position = []
        self.df = None

    def step(self, action):
        info = {}
        if self.done:
            return np.zeros(self.num_states), 0, self.done, {}

        if isinstance(self.frameskip, int):
            self.num_steps = self.frameskip
        else:
            self.num_steps = np.random.randint(self.frameskip[0], self.frameskip[1]+1)

        reward = self._take_action(action)

        self.refresh_max_wallet(action, reward)

        obs = self.getState()

        if (self.cumsum - self.max_wallet) <= np.log(0.95):
            self.done = True
        if self.cursor >= len(self.df):
            self.done = True
        elif self.ep >= self.max_ep:
            self.done = True

        if self.done:
            self.ep_mean = 0.99 * self.ep_mean + 0.01 * self.ep
            # print("{}/{:.1f}, {:.3f} / {:.3f}".format(self.ep, self.ep_mean, np.exp(self.cumsum), np.exp(self.max_wallet)))

        return obs, reward, self.done, {}

    def refresh_max_wallet(self, action, reward):
        self.cumsum += reward
        if self.max_wallet < self.cumsum:
            self.max_wallet = self.cumsum
        self.last_action = action

    def reset(self):
        if self.mode=="train":
            self.start_point = np.random.randint(0, self.df_size - self.max_ep + 1)
        else:
            self.start_point = 0
        self.cursor = 0
        self.col = 'btc_adjusted'
        self.df = pd.DataFrame(list(self.collection.find(filter={"_id":
                                                                       {"$gte":
                                                                            self.start_point}
                                                                   },
                                                         sort=[("_id", 1)],
                                                         limit=min(int(self.ep_mean), self.max_ep)+1024)
                                    )
                               )
        self.last_action = 0
        self.ep = 0
        self.max_wallet = 0.0
        self.cumsum = 0.0
        self.done = False

        state = self.getState()
        return state

    def _take_action(self, action):
        last_signal = [0, 1, -1][self.last_action]
        signal = [0, 1, -1][action]
        gap = np.sum(self.df.iloc[self.cursor:self.cursor+self.num_steps][self.col].values)
        reward = 0
        reward += self.fee * abs(last_signal - signal)
        reward += signal * gap
        reward = np.nan_to_num(reward)
        r_reward = self.lever * (np.exp(reward) - 1) + 1
        if r_reward <= 0.001:
            r_reward = 0.001
        lever_reward = np.log(r_reward)

        return lever_reward

    def seed(self, seed=None):
        random.seed(seed)

    def getState(self):
        if isinstance(self.frameskip, int):
            self.num_steps = self.frameskip
        else:
            self.num_steps = np.random.randint(self.frameskip[0], self.frameskip[1]+1)

        self.last_cursor = self.cursor
        self.cursor += self.num_steps
        self.ep += self.num_steps

        def ohlc(x):
            _open = x[0, 0]
            _high = np.max(x[:, 1], axis=0)
            _low = np.min(x[:, 2], axis=0)
            _close = x[-1, 3]
            return np.array([_open, _high, _low, _close]).T.flatten()

        price_value = np.log(ohlc(self.df.iloc[self.last_cursor:self.cursor][self.columns].values))
        diff_matrix = np.subtract.outer(price_value, price_value)
        self.state = diff_matrix[np.triu_indices(4, k = 1)]
        # current_state = self.df.iloc[self.last_cursor:self.cursor][self.columns].values.mean(0)
        # obs = np.array([self.last_state, current_state]).flatten()

        return self.state

    def render(self, mode='human', close=False):
        pass
