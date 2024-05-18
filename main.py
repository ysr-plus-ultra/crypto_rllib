import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
import logging
import numpy as np
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from custom.impala_custom import ImpalaConfig
import time
from datetime import datetime
# env setup
import copy
from crypto_env import CryptoEnv
from ray.tune.registry import register_env
from pprint import pprint
env_cfg = {
    "NUM_STATES": 4,
    "NUM_ACTIONS": 3,

    "FEE": 0.1,
    "MAX_EP": 15000,
    "DF_SIZE": 111260,
    "frameskip": 5,
    "mode": "train",
}
# load_model_path = "/checkpoint/model_eval_32_copy"
load_model_path = None
new_model_path = "/checkpoint/model_20240509"
eval_model_path = "/checkpoint/model_eval_32"

# model setup
from custom.custom_model import CustomRNNModel
from ray.rllib.models import ModelCatalog
import ray
ray.init(log_to_driver=False)

ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
# model setup end
num_env_workers = 4
num_env = 16
num_rollout = 64
config = ImpalaConfig()

config = config.training(gamma=0.5,
                         lr=1e-3,
                         train_batch_size=num_rollout*num_env*num_env_workers//2,
                         model={
                             "custom_model": "my_torch_model",
                             "lstm_use_prev_action": False,
                             "lstm_use_prev_reward": False,
                             "custom_model_config": {"NUM_STATES": env_cfg["NUM_STATES"],
                                                     "fc_size": 4,
                                                     "lstm_size": 8,
                                                     "hidden_size": 64},
                             "max_seq_len": num_rollout,
                         },
                         vtrace=True,
                         opt_type="rmsprop",
                         entropy_coeff=0.01,
                         vf_loss_coeff=0.5,
                         momentum=0.0,
                         epsilon=1e-08,
                         decay=0.0, #1e-6
                         # grad_clip=1.0,
                         # grad_clip_by="global_norm",
                         # replay_proportion = 0.25,
                         # replay_buffer_num_slots = 1024,
                         )

config = config.framework(framework="torch")
config = config.resources(num_gpus = 1.0,
                          num_gpus_per_worker=0.5/num_env_workers/2
                          )
config = config.environment(env = "my_env", env_config=env_cfg)
config = config.exploration(exploration_config = {"type": "StochasticSampling", "random_timesteps": 1e6},)
config = config.rollouts(num_rollout_workers=num_env_workers,
                         num_envs_per_worker=num_env,
                         rollout_fragment_length=num_rollout,)

eval_config = copy.deepcopy(env_cfg)
eval_config["MAX_EP"] = 8955
eval_config["DF_SIZE"] = 8955
eval_config["mode"] = "eval"
eval_config["FEE"] = 0.05
def env_creator(env_config):
    return CryptoEnv(env_config)
register_env("my_env", env_creator)

config = config.evaluation(evaluation_interval= 5,
                           evaluation_duration = 128,
                           evaluation_duration_unit = "episodes",
                           evaluation_config = {"env": "my_env" ,
                                                "env_config":eval_config,
                                                "num_envs_per_worker": 4,
                                                "exploration_config" : {
                                                    "type": "StochasticSampling",
                                                    "random_timesteps": 0
                                                },
                                                "num_gpus":0,
                                                "num_gpus_per_worker": 0
                                                },
                           evaluation_parallel_to_training = True,
                           evaluation_num_workers = 8)
algo = config.build()
if load_model_path is not None:
    algo.restore(load_model_path)
last_time = time.time()
target_metric = -1.0
average_weight = 0.9
max_metric = 0.0
ma_benchmark = 0.0
while 1:
    result = algo.train()
    current_time = time.time()

    try:
        frameskip = np.sqrt(43200/env_cfg["frameskip"])
        benchmark = np.nan_to_num(result["evaluation"]["episode_reward_mean"])/frameskip
        ma_benchmark = (average_weight * ma_benchmark) + ((1-average_weight)*benchmark)
        if ma_benchmark > max_metric:
            max_metric = ma_benchmark
            print(datetime.fromtimestamp(current_time), f"eval: {max_metric:.4f} = {max_metric*frameskip:.4f} / {frameskip:.4f}")
            algo.save(eval_model_path)
    except:
        pass

algo.save(new_model_path)
ray.shutdown()