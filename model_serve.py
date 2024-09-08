import ray
from typing import Dict
import json
from custom.impala_custom import ImpalaConfig
from ray import serve
from ray.rllib.models import ModelCatalog
from custom.custom_model import CustomRNNModel
from starlette.requests import Request
from ray.rllib.utils.framework import try_import_torch
from crypto_env import CryptoEnv
from ray.tune.registry import register_env
torch, nn = try_import_torch()
from ray.serve.schema import LoggingConfig
from collections import defaultdict
import numpy as np
ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
# model setup end
def env_creator(env_config):
    return CryptoEnv(env_config)
register_env("TradingEnv", env_creator)
eval_model_path = "/checkpoint/model_eval_256"
env_cfg = {
    "FEE": 0.1,
    "MAX_EP": 8000,
    "DF1_SIZE": 115818,
    "DF2_SIZE": 5831,
    "DF3_SIZE": 7376,
    "MODE": "train",
}
lstm_size = 128

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def json_zip(j):
    j = json.dumps(j, cls=NumpyArrayEncoder).encode('utf-8')
    return j

def json_unzip(j, insist=True):
    j = json.loads(j)
    return j

@serve.deployment(
    route_prefix="/model",
    ray_actor_options={"num_gpus": 1},
    logging_config=LoggingConfig(log_level="WARN"),)
class ServeModel:
    def __init__(self, checkpoint_path):
        impala_config = ImpalaConfig()
        impala_config = (impala_config.
                         training(gamma=0.0, lr=0.001, train_batch_size=500,
                                               model={
                                                   "custom_model": "my_torch_model",
                                                   "lstm_use_prev_action": False,
                                                   "lstm_use_prev_reward": False,
                                                   "custom_model_config": {"encoder_size": 4,
                                                                           "hidden_size": 512,
                                                                           "cell_size": 32,
                                                                           "popart_beta": 1e-3},
                                               },
                                               )
                         .framework(framework="torch")
                         .environment(env="TradingEnv",
                                      env_config=env_cfg,)
                         .resources(num_gpus=1)
                         .env_runners(num_env_runners=0)
                         )

        self.algo = impala_config.build()
        self.algo.restore(checkpoint_path)
        self.policy = self.algo.get_policy()
        self.state = defaultdict(lambda: [np.zeros(lstm_size), np.zeros(lstm_size)])

    async def __call__(self, starlette_request: Request) -> Dict:
        request = await starlette_request.body()
        request = request.decode("utf-8")
        request = json.loads(request)

        action, state_out, _ = self.algo.compute_actions(
            observations=request,
            state=self.state
        )

        for k,v in state_out.items():
            self.state[k] = v
        return action

# Defining the builder function. This is so we can start our deployment via:
# `serve run [this py module]:rl_module checkpoint=[some algo checkpoint path]`
def rl_module(args: Dict[str, str]):
    return ServeModel.bind(args["checkpoint"])