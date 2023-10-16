"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python main.py
For CLI options:
$ python main.py --help
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
import argparse

import ray
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from custom_model import CustomRNNModel
from impala_custom import ImpalaConfig
# from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv,NoopResetEnv,FireResetEnv
import gym
import time
import psutil, gc
from ray.tune.registry import register_env
env_config = {
    "NUM_STATES": 13,
    "NUM_ACTIONS": 3,

    "FEE": 0.04,
    "MAX_EP": 14400,
    "DF_SIZE": 172800, #

    "frameskip": 5,
    "mode": "train",
}

from crypto_env import CryptoEnv
torch, nn = try_import_torch()
torch.backends.cudnn.benchmark = True
def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
    impala_config = ImpalaConfig()
    impala_config = impala_config.training(gamma=0.0, lr=3e-4, train_batch_size=2048,
                                           model = {
                                               "custom_model": "my_torch_model",
                                               "lstm_use_prev_action": True,
                                               "lstm_use_prev_reward": False,
                                               "custom_model_config": {
                                               },
                                           },
                                           vtrace=True,
                                           vtrace_drop_last_ts = False,
                                           opt_type = "rmsprop",
                                           entropy_coeff= 0.001,
                                           vf_loss_coeff = 1.0,
                                           momentum = 0.0,
                                           epsilon= 1e-08,
                                           decay = 0.0,
                                           ) \
        .framework(framework="torch") \
        .environment(env = CryptoEnv, env_config= env_config, disable_env_checking=True,)\
        .rollouts(num_rollout_workers=8, num_envs_per_worker=8, rollout_fragment_length=32) \
        # .environment(env = CryptoEnv, env_config= env_config, disable_env_checking=True,)\
        # .environment(env="LunarLander-v2", disable_env_checking=True, ) \
    # .environment(env = "pong_env",disable_env_checking=True,)\

    # pprint(impala_config.to_dict())
    ray.init()

    algo = impala_config.build()
    # algo.restore("D:\checkpoint\checkpoint_000568")
    # policy = trainer.get_policy()
    last_time = time.time()
    # algo.save("/checkpoint/init")
    while 1:
        auto_garbage_collect()
        result = algo.train()
        current_time = time.time()
        # print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        # if (result["episode_reward_mean"] >= args.stop_reward
        # ):
        #     break
        # if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
        if (current_time - last_time) > 300:
            algo.save("/checkpoint/")
            last_time = current_time
    ray.shutdown()