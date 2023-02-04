from crypto_env import CryptoEnv
import unittest

env_config = {
    "NUM_STATES": 16,
    "NUM_ACTIONS": 3,

    "FEE": 0.05,
    "LEVERAGE": 1.0,
    "MAX_EP": 11520,

    "frameskip": (1, 5),
    "mode": "train",
    "col": 'btc_adjusted',
    "worker_index": 0,
    "num_workers":1,
    "vector_env_index":1,
    "df_size":11520
}
env = CryptoEnv(config=env_config)

class TestStringMethods(unittest.TestCase):

    def test_step_0(self):
        env.reset()
        done = False
        rr = 0.0
        while not done:
            _,r,done,_ = env.step(0)
            rr+=r

        self.assertAlmostEqual(rr, 0.0)

    def test_step_1(self):
        env.reset()
        done = False
        rr = 0.0
        while not done:
            _, r, done, _ = env.step(1)
            rr += r

        self.assertAlmostEqual(rr, 0.0)

    def test_step_2(self):
        env.reset()
        done = False
        rr = 0.0
        while not done:
            _, r, done, _ = env.step(2)
            rr += r

        self.assertAlmostEqual(rr, 0.0)
    def test_random_step(self):
        env.reset()
        env.step(0)
        self.assertEqual(env._gap_period0, env._state_period0)
        self.assertEqual(env._gap_period1, env._state_period1)

if __name__ == '__main__':
    unittest.main()