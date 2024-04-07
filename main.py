import warnings
import logging
import numpy as np
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from custom.impala_custom import ImpalaConfig
import time
from datetime import datetime
# env setup
from crypto_env import CryptoEnv
from ray.tune.registry import register_env
from pprint import pprint
env_cfg = {
    "NUM_STATES": 4,
    "NUM_ACTIONS": 3,

    "FEE": 0.07,
    "MAX_EP": 10800,
    "DF_SIZE": 1038240,

    "frameskip": 15,
    "mode": "train",
}
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
num_rollout_worker = 8
num_env = 16
num_rollout = 32
config = ImpalaConfig()

config = config.training(gamma=0.99, lr=1e-3, train_batch_size=32,
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
                                           epsilon=1e-6,
                                           decay=0.0,
                                           grad_clip=1.0,
                                           grad_clip_by="norm",
                                           )

config = config.framework(framework="torch")
config = config.resources(num_gpus=0.5,
                          )
config = config.environment(env = "my_env", env_config=env_cfg)
config = config.exploration(exploration_config = {"type": "StochasticSampling"},)
config = config.rollouts(num_rollout_workers=num_rollout_worker,
                         create_env_on_local_worker=True,
                         num_envs_per_worker=num_env,
                         rollout_fragment_length=num_rollout)
algo = config.build()
# algo.restore("/checkpoint/model_rnn_1024_tf_60_fee_0_07")
last_time = time.time()
target_metric = -1.0
average_weight = 0.9
max_metric = 0.0
while 1:
    result = algo.train()
    current_time = time.time()
    # stop training of the target train steps or reward are reached
    try:
        benchmark = result["info"]["learner"]["default_policy"]["learner_stats"]["vf_explained_var"]
        target_metric = average_weight * target_metric + (1-average_weight) * np.nan_to_num(benchmark)
        if target_metric > 0.9:
            break
        if result["info"]["learner"]["default_policy"]["learner_stats"]["var_gnorm"] > 1e4:
            break
    except:
        pass

    # if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
    if target_metric > max_metric*1.01:
        algo.save("/checkpoint/model_20240407")
        max_metric = target_metric

        if (current_time - last_time) > 600:
            last_time = current_time
            print(datetime.fromtimestamp(current_time), "{:.4f}".format(target_metric))

ray.shutdown()