import warnings
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
    "NUM_STATES": 13,
    "NUM_ACTIONS": 3,

    "FEE": 0.06,
    "MAX_EP": 129600,
    "DF_SIZE": 1036800,
    # "DF_SIZE": 172800,
    "frameskip": 30,
    "mode": "train",
}
# load_model_path = "/checkpoint/model_eval_8"
load_model_path = None
new_model_path = "/checkpoint/model_20240428"
eval_model_path = "/checkpoint/model_eval_32"
def env_creator(env_config):
    return CryptoEnv(env_config)
register_env("my_env", env_creator)

# env setup end

# model setup
from custom.custom_model import CustomRNNModel
from ray.rllib.models import ModelCatalog
import ray
ray.init(log_to_driver=False)

ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
# model setup end
num_env_workers = 8
num_env = 4
num_rollout = 64
config = ImpalaConfig()

config = config.training(gamma=0.0, lr=1e-3, train_batch_size=1024,
                                           model={
                                               "custom_model": "my_torch_model",
                                               "lstm_use_prev_action": True,
                                               "lstm_use_prev_reward": True,
                                               "custom_model_config": {
                                               },
                                               "max_seq_len": num_rollout,
                                           },
                                           vtrace=True,
                                           opt_type="rmsprop",
                                           entropy_coeff=0.01,
                                           vf_loss_coeff=0.5,
                                           momentum=0.0,
                                           epsilon=1e-08,
                                           decay=0.0,
                                           grad_clip=1.0,
                                           grad_clip_by="global_norm",
                                           )

config = config.framework(framework="torch")
config = config.resources(num_gpus = 0.5,
                          )
config = config.environment(env = "my_env", env_config=env_cfg)
config = config.exploration(exploration_config = {"type": "StochasticSampling"},)
config = config.rollouts(num_rollout_workers=num_env_workers,
                         num_envs_per_worker=num_env,
                         rollout_fragment_length=512,)

eval_config = copy.deepcopy(env_cfg)
eval_config["MAX_EP"] = 43200
eval_config["DF_SIZE"] = 43200
eval_config["mode"] = "eval"
eval_config["FEE"] = 0.05

config = config.evaluation(evaluation_interval= 10,
                           evaluation_duration = 32,
                           evaluation_duration_unit = "episodes",
                           evaluation_config = {"env": "my_env" ,
                                                "env_config":eval_config,
                                                "num_envs_per_worker": 4},
                           evaluation_parallel_to_training = True,
                           evaluation_num_workers = 8)
algo = config.build()
if load_model_path is not None:
    algo.restore(load_model_path)
last_time = time.time()
target_metric = -1.0
average_weight = 0.5
max_metric = 0.0
eval_metric = 0.0
while 1:
    result = algo.train()
    current_time = time.time()
    # stop training of the target train steps or reward are reached
    # try:
    #     benchmark = result["info"]["learner"]["default_policy"]["learner_stats"]["vf_explained_var"]
    #     target_metric = average_weight * target_metric + (1-average_weight) * np.nan_to_num(benchmark)
    #     if target_metric > 0.9:
    #         break
    #     if result["info"]["learner"]["default_policy"]["learner_stats"]["var_gnorm"] > 1e4:
    #         break
    # except:
    #     pass

    # if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
    # if (current_time - last_time) > 600:
    #     algo.save(new_model_path)
    #     max_metric = target_metric
    #     last_time = current_time
    #     print(datetime.fromtimestamp(current_time), "{:.4f}".format(target_metric))

    try:
        eval_benchmark = np.nan_to_num(result["evaluation"]["episode_reward_mean"])/np.sqrt(43200/30)
        eval_smooth = average_weight * eval_metric + (1-average_weight) * eval_benchmark
        if eval_smooth > eval_metric:
            eval_metric = eval_smooth
            print(datetime.fromtimestamp(current_time), "eval: {:.4f}".format(eval_smooth))
            algo.save(eval_model_path)
    except:
        pass

algo.save(new_model_path)
ray.shutdown()