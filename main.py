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
import argparse

import ray
from ray.rllib.agents import impala
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from custom_model import CustomRNNModel
from impala_custom import ImpalaCustomTrainer
import time
env_config = {
    "NUM_STATES": 6,
    "NUM_ACTIONS": 3,

    "FEE": 0.05,
    "LEVERAGE": 1.0,
    "MAX_EP": 10080,

    "frameskip": (1, 15),
    "mode": "train",
    "col": 'btc_adjusted'
}
from pymongo import MongoClient
client = MongoClient("mongodb://admin:dbstnfh123@192.168.0.201:27017")
db = client.Binance
collection = db.Binance
df_size = collection.count_documents({})
env_config['df_size'] = df_size
client.close()
from crypto_env import CryptoEnv
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="IMPALA", help="The RLlib-registered algorithm to use."
)
# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: --stop-reward must "
#     "be achieved within --stop-timesteps AND --stop-iters.",
# )
# parser.add_argument(
#     "--stop-iters", type=int, default=50, help="Number of iterations to train."
# )
# parser.add_argument(
#     "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
# )
parser.add_argument(
    "--stop-reward", type=float, default=1.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
#
# class TorchCustomModel(TorchModelV2, nn.Module):
#     """Example of a PyTorch custom model that just delegates to a fc-net."""
#
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name
#         )
#         nn.Module.__init__(self)
#
#         self.torch_sub_model = TorchFC(
#             obs_space, action_space, num_outputs, model_config, name
#         )
#
#     def forward(self, input_dict, state, seq_lens):
#         input_dict["obs"] = input_dict["obs"].float()
#         fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
#         return fc_out, []
#
#     def value_function(self):
#         return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")


    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    # ModelCatalog.register_custom_model(
    #     "my_model", TorchCustomModel
    # )
    ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
    config = {
        "disable_env_checking": True,
        "env": CryptoEnv,
        "env_config": env_config,
        "gamma": 0.0,
        "grad_clip": None,
        "model": {
            "custom_model": "my_torch_model",
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "lstm_cell_size": 6,

            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
            },
        },
        "vtrace_drop_last_ts": True,
        "num_workers": 4,  # parallelism
        "framework": "torch",
        "rollout_fragment_length": 50,
        "train_batch_size": 500,
        "opt_type" : "rmsprop",
        "num_envs_per_worker" : 4,
        "entropy_coeff": 0.001,
        "momentum": 0.0,
        "epsilon": 1e-8,
        "decay": 0.0,
        "num_gpus": 0.5,

    }

    stop = {
        "episode_reward_mean": args.stop_reward,
    }


    ray.init(local_mode=args.local_mode)
    impala_config = impala.DEFAULT_CONFIG.copy()
    impala_config.update(config)
    impala_config["lr"] = 1e-3
    trainer = ImpalaCustomTrainer(config=impala_config)
    # policy = trainer.get_policy()
    last_time = time.time()
    trainer.save("/checkpoint/init")
    while 1:
        result = trainer.train()
        current_time = time.time()
        # print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (result["episode_reward_mean"] >= args.stop_reward
        ):
            break
        if (result["episode_reward_mean"] >= 0 and (current_time-last_time)>300):
            trainer.save("/checkpoint/")
            last_time = current_time
    trainer.save("/checkpoint/final_model")
    ray.shutdown()