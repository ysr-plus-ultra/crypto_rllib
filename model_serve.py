import ray

from custom.impala_custom import ImpalaConfig
from ray import serve
from ray.rllib.models import ModelCatalog
from custom.custom_model import CustomRNNModel
from starlette.requests import Request
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import os
import zlib, json, base64
from collections import defaultdict
ZIPJSON_KEY = 'base64(zip(o))'
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
torch.backends.cudnn.benchmark = True
ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
_action_space = Discrete(3)
_observation_space = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
_lstm_size = 1024
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

@serve.deployment(route_prefix="/model", ray_actor_options={"num_gpus": 1})
class ServeModel:
    def __init__(self, checkpoint_path):
        try:
            impala_config = ImpalaConfig()
            impala_config = impala_config.training(gamma=0.0, lr=0.001, train_batch_size=500,
                                                   model={
                                                       "custom_model": "my_torch_model",
                                                       "lstm_use_prev_action": True,
                                                       "lstm_use_prev_reward": True,
                                                       "custom_model_config": {
                                                       },
                                                   },
                                                   ) \
                .framework(framework="torch") \
                .environment(observation_space=_observation_space, action_space=_action_space,
                             disable_env_checking=True, ) \
                .resources(num_gpus=1,
                          )\
                .rollouts(num_rollout_workers=0)
            self.algorithm = impala_config.build()
            self.algorithm.restore(checkpoint_path)
            self._policy = self.algorithm.workers.local_worker().get_policy()
            self.state_h = defaultdict(lambda: np.zeros(_lstm_size))
            self.state_c = defaultdict(lambda: np.zeros(_lstm_size))

        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

    async def __call__(self, request: Request):
        compressed_input = await request.body()
        json_input = json_unzip(compressed_input)

        obs = [np.array(x) for x in json_input["observation"]]
        prev_a = json_input["prev_a"]
        prev_r = json_input["prev_r"]
        magic_number = json_input["magic_number"]
        state_h = np.array([self.state_h[x] for x in magic_number])
        state_c = np.array([self.state_c[x] for x in magic_number])

        action, state_out, _ = self._policy.compute_actions(
            obs_batch=np.array(obs),
            state_batches=[state_h, state_c],
            prev_action_batch=prev_a,
            prev_reward_batch=prev_r)

        value = self._policy.model.value_function()
        compressed_output = json_zip({"action": action, "value": value.to('cpu').detach().numpy() })
        for x, s_h, s_c in zip(magic_number, state_out[0], state_out[1]):
            self.state_h[x] = s_h
            self.state_c[x] = s_c
        return compressed_output

if __name__ == "__main__":
    try:
        ray.init(address="auto", namespace="serve")
        serve.start(detached=True)
        impala_model = ServeModel.bind("D:\checkpoint\model_rnn_1024_tf_60_fee_0_07")
        serve.run(impala_model)
    finally:
        ray.shutdown()