import warnings

import numpy as np

warnings.filterwarnings('ignore')
from custom.impala_custom import ImpalaConfig
import time
# env setup
from crypto_env import CryptoEnv
from ray.tune.registry import register_env
env_cfg = {
    "NUM_STATES": 7,
    "NUM_ACTIONS": 3,

    "FEE": 0.1,
    "MAX_EP": 43200,
    "DF_SIZE": 260640,  #
    # "DF_SIZE": 131040,  #

    "frameskip": 5,
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
# ray.init(object_store_memory=4 * 1024 * 1024 * 1024)

ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
# model setup end
config = ImpalaConfig()
config = config.training(gamma=0.5, lr=1e-3, train_batch_size=256,
                                           model={
                                               "custom_model": "my_torch_model",
                                               "lstm_use_prev_action": True,
                                               "lstm_use_prev_reward": False,
                                               "custom_model_config": {
                                               },
                                           },
                                           vtrace=True,
                                           opt_type="rmsprop",
                                           entropy_coeff=0.01,
                                           vf_loss_coeff=0.5,
                                           momentum=0.0,
                                           epsilon=1e-08,
                                           decay=0.0,
                                           grad_clip=1.0,
                                           )
config = config.framework(framework="torch")
config = config.resources(num_gpus=0.5)
config = config.environment(env = "my_env", env_config=env_cfg)
config = config.rollouts(num_rollout_workers=8, num_envs_per_worker=4, rollout_fragment_length=32)
algo = config.build()
# algo.restore("/checkpoint/model_20240208")
last_time = time.time()
target_metric = 0.0
average_weight = 0.9
while 1:
    result = algo.train()
    current_time = time.time()
    # print(pretty_print(result))
    # stop training of the target train steps or reward are reached
    target_metric = average_weight * target_metric + (1-average_weight) * np.nan_to_num(result["episode_reward_mean"])
    if (target_metric >= 0.7
    ):
        break
    # if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
    if (current_time - last_time) > 600:
        algo.save("/checkpoint/model_20240227")
        last_time = current_time
        print(target_metric)

ray.shutdown()