import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

from warnings import filterwarnings
filterwarnings("ignore")

import ray
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from custom.custom_model import CustomRNNModel
from custom.impala_custom import ImpalaConfig
from ray import tune
# from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env

env_cfg = {
    "NUM_STATES": 2,
    "NUM_ACTIONS": 3,

    "FEE": 0.06,
    "MAX_EP": 21600,
    "DF_SIZE": 864000,  #

    "frameskip": 5,
    "mode": "train",
}

from crypto_env import CryptoEnv

torch, nn = try_import_torch()
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    register_env("my_env", lambda config: CryptoEnv(env_cfg))
    ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
    impala_config = ImpalaConfig()
    impala_config = impala_config.training(gamma=0.5, lr=1e-3, train_batch_size=512,
                                           model={
                                               "custom_model": "my_torch_model",
                                               "lstm_use_prev_action": True,
                                               "lstm_use_prev_reward": False,
                                               "custom_model_config": {
                                               },
                                           },
                                           vtrace=True,
                                           opt_type="rmsprop",
                                           entropy_coeff=0.001,
                                           vf_loss_coeff=1.0,
                                           momentum=0.0,
                                           epsilon=1e-08,
                                           decay=0.0,
                                           grad_clip=0.0,
                                           ) \
        .framework(framework="torch") \
        .environment(env="my_env", env_config=env_cfg, disable_env_checking=True) \
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2, rollout_fragment_length=64) \
        # .environment(env = CryptoEnv, env_config= env_config, disable_env_checking=True,)\
    # .environment(env="LunarLander-v2", disable_env_checking=True, ) \
    # .environment(env = "pong_env",disable_env_checking=True,)\

    # pprint(impala_config.to_dict())
    ray.init(object_store_memory=4 * 1024 * 1024 * 1024)
    tune.Tuner("IMPALA")
    algo = impala_config.build(env = "my_env", env_config=env_cfg)
    algo.train()
    # algo.restore("D:\modelbackup\checkpoint_002512")
    # policy = trainer.get_policy()
    last_time = time.time()
    # algo.save("/checkpoint/init")
    while 1:
        result = algo.train()
        current_time = time.time()
        # print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        # if (result["episode_reward_mean"] >= args.stop_reward
        # ):
        #     break
        # if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
        if (current_time - last_time) > 600:
            algo.save("/checkpoint/")
            last_time = current_time
    ray.shutdown()
