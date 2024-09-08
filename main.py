import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

def main():
    import numpy as np
    import time
    import copy
    from datetime import datetime

    from custom.impala_custom import ImpalaConfig
    from crypto_env import CryptoEnv
    from ray.tune.registry import register_env

    # model setup
    from custom.custom_model import CustomRNNModel
    from ray.rllib.models import ModelCatalog
    import ray

    env_cfg = {
        "FEE": 0.075,
        "MAX_EP": 8000,
        "DF1_SIZE": 115693,
        "DF2_SIZE": 6842,
        "DF3_SIZE": 6579,
        "MODE": "train",
    }
    # load_model_path = "/checkpoint/model_20240529"
    load_model_path = None
    new_model_path = "/checkpoint/model_2024-09-08"
    eval_model_path = "/checkpoint/model_eval_512"

    ray.init(log_to_driver=False)

    ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
    # model setup end

    def env_creator(env_config):
        return CryptoEnv(env_config)

    register_env("TradingEnv",  env_creator)

    num_env_workers = 8
    num_env = 32
    num_rollout = 64

    config = ImpalaConfig()

    config = config.training(gamma=0.5,
                             lr=1e-2,
                             train_batch_size=num_env_workers * num_env * num_rollout,
                             model={
                                 "custom_model": "my_torch_model",
                                 "lstm_use_prev_action": False,
                                 "lstm_use_prev_reward": False,
                                 "custom_model_config": {
                                     "encoder_size": 4,
                                     "hidden_size": 512,
                                     "cell_size": 256,
                                     "popart_beta": 1e-3
                                 },
                                 "max_seq_len": num_rollout,
                             },
                             vtrace=True,
                             opt_type="rmsprop",
                             entropy_coeff=1e-2,
                             vf_loss_coeff=0.5,
                             momentum=0.0,
                             epsilon=1e-6,
                             decay=0.0,  # 1e-6
                             grad_clip=1.0,
                             grad_clip_by="global_norm",
                             )
    config = config.environment(env="TradingEnv",
                                env_config=env_cfg)
    config = config.framework(framework="torch")
    config = config.resources(num_gpus=1, )
    config = config.env_runners(num_env_runners=num_env_workers,
                                num_envs_per_env_runner=num_env,
                                rollout_fragment_length=num_rollout,)

    eval_config = copy.deepcopy(env_cfg)
    eval_config["MODE"] = "eval"
    eval_config["FEE"] = 0.055

    config = config.evaluation(evaluation_interval=10,
                               evaluation_duration=128,
                               evaluation_duration_unit="episodes",
                               # evaluation_force_reset_envs_before_iteration=True,
                               evaluation_config={
                                   "env": "TradingEnv",
                                   "env_config": eval_config,
                                   "num_envs_per_env_runner": 16,
                               },
                               evaluation_num_env_runners=8,
                               evaluation_parallel_to_training = True
                               )
    algo = config.build()
    if load_model_path is not None:
        algo.restore(load_model_path)
    last_time = time.time()
    target_metric = -1.0
    average_weight = 0.95
    max_metric = 0.0
    ma_benchmark = 0.0
    while 1:
        result = algo.train()
        current_time = time.time()

        try:
            frameskip = np.sqrt(43200)
            benchmark = np.nan_to_num(result["evaluation"]["env_runners"]["episode_reward_mean"]) / frameskip
            ma_benchmark = (average_weight * ma_benchmark) + ((1 - average_weight) * benchmark)
            if ma_benchmark > max_metric:
                max_metric = ma_benchmark
                print(datetime.fromtimestamp(current_time),
                      f"eval: {benchmark:.4f} ({max_metric:.4f}) = {max_metric * frameskip:.4f} / {frameskip:.4f}")
                algo.save(eval_model_path)
        except:
            pass

    algo.save(new_model_path)
    ray.shutdown()


if __name__ == '__main__':
    main()
