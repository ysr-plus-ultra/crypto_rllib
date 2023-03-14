from crypto_env import CryptoEnv
import unittest

env_config = {
    "NUM_STATES": 10,
    "NUM_ACTIONS": 3,

    "FEE": 0.0,
    "MAX_EP": 4320,
    "DF_SIZE": 4320,

    "frameskip": (3,7),
    "mode": "train",
    "col": 'btc_adjusted'
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

if __name__ == '__main__':
    unittest.main()