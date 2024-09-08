#!/usr/bin/env python
import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import numpy as np
import gymnasium as gym
import time
from datetime import datetime
import logging
import ray
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import os
from crypto_env import CryptoEnv
from custom.impala_custom import ImpalaConfig
from custom.custom_model import CustomRNNModel

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900
env_cfg = {
    "NUM_STATES": 8,
    "NUM_ACTIONS": 3,

    "FEE": 0.1,
    "MAX_EP": 12000,
    "DF_SIZE": 186982,
    "frameskip": 3,
    "mode": "train",
}
def env_creator(env_config):
    return CryptoEnv(env_config)
register_env("my_env", env_creator)
# load_model_path = "/checkpoint/model_eval_32"
load_model_path = None
new_model_path = "/checkpoint/model_20240509"
eval_model_path = "/checkpoint/model_eval_32"


ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
num_env_workers = 4
num_env = 0
num_rollout = 64
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    ray.init(log_to_driver=False)

    # Algorithm config. Note that this config is sent to the client only in case
    # the client needs to create its own policy copy for local inference.
    config = (
        ImpalaConfig()
        .training(gamma=0.0,
                  lr=1e-3,
                  train_batch_size=32,
                  model={
                      "custom_model": "my_torch_model",
                      "lstm_use_prev_action": False,
                      "lstm_use_prev_reward": False,
                      "custom_model_config": {"NUM_STATES": env_cfg["NUM_STATES"],
                                              "fc_size": 16,
                                              "lstm_size": 8,
                                              "hidden_size": 64},
                      "max_seq_len": num_rollout,
                  },
                  vtrace=True,
                  opt_type="rmsprop",
                  entropy_coeff=0.001,
                  vf_loss_coeff=0.5,
                  momentum=0.0,
                  epsilon=1e-08,
                  decay=0.0,  # 1e-6
                  grad_clip=1.0,
                  grad_clip_by="global_norm",
                  # replay_proportion = 1.0,
                  # replay_buffer_num_slots = 512,
                  )
        # Indicate that the Algorithm we setup here doesn't need an actual env.
        # Allow spaces to be determined by user (see below).
        .environment(env = "my_env", env_config=env_cfg)
        # DL framework to use.
        .framework(framework="torch")
        # Use the `PolicyServerInput` to generate experiences.
        .offline_data(input_=lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, SERVER_BASE_PORT),
                      input_config = {"env": "my_env", "env_config": env_cfg}
                      )
        # Use n worker processes to listen on different ports.
        .resources(num_gpus=1,
                   num_cpus_per_worker = 4,
                   num_gpus_per_worker = 0.1
                   )
        .rollouts(num_rollout_workers=0,
                  rollout_fragment_length=num_rollout,
                  enable_connectors=False,
                  )
        # Disable OPE, since the rollouts are coming from online clients.
        .evaluation(off_policy_estimation_methods={})
        # Set to INFO so we'll see the server's actual address:port.
    )
    config.experimental(_enable_new_api_stack=False)
    # eval_config = copy.deepcopy(env_cfg)
    # eval_config["MAX_EP"] = 7467
    # eval_config["DF_SIZE"] = 7467
    # eval_config["mode"] = "eval"
    # eval_config["FEE"] = 0.05
    #
    # config.evaluation(evaluation_interval=10,
    #                   evaluation_duration=128,
    #                   evaluation_duration_unit="episodes",
    #                   evaluation_config={"env": "my_env",
    #                                      "env_config": eval_config,
    #                                      "num_envs_per_worker": 4,
    #                                      "exploration_config": {
    #                                          "type": "StochasticSampling",
    #                                          "random_timesteps": 0
    #                                      },
    #                                      "num_gpus": 0,
    #                                      "num_gpus_per_worker": 0,
    #                                      "num_gpus_per_learner_worker": 0,
    #                                      },
    #                   evaluation_parallel_to_training=True,
    #                   evaluation_num_workers=8)
    algo = config.build()

    if load_model_path is not None:
        algo.restore(load_model_path)
    target_metric = -1.0
    average_weight = 0.5
    max_metric = 0.0
    eval_metric = 0.0

    print("algo successfully loaded")
    while 1:
        result = algo.train()
        current_time = time.time()
        try:
            eval_benchmark = np.nan_to_num(result["evaluation"]["episode_reward_mean"]) / np.sqrt(
                43200 / env_cfg["frameskip"])
            if eval_benchmark > eval_metric:
                eval_metric = eval_benchmark
                print(datetime.fromtimestamp(current_time), "eval: {:.4f}".format(eval_benchmark))
                algo.save(eval_model_path)
        except:
            pass

    algo.save(new_model_path)
    algo.stop()
