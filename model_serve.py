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
ZIPJSON_KEY = 'base64(zip(o))'
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
torch.backends.cudnn.benchmark = True
ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
_action_space = Discrete(3)
_observation_space = Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def json_zip(j):
    j = base64.b64encode(
            zlib.compress(
                json.dumps(j, cls=NumpyArrayEncoder).encode('utf-8')
            )
        )

    return j

def json_unzip(j, insist=True):
    j = zlib.decompress(base64.b64decode(j))
    j = json.loads(j)
    return j

@serve.deployment(route_prefix="/model", ray_actor_options={"num_gpus": 0.5})
class ServeModel:
    def __init__(self, checkpoint_path):
        try:
            impala_config = ImpalaConfig()
            impala_config = impala_config.training(gamma=0.0, lr=0.001, train_batch_size=500,
                                                   model={
                                                       "custom_model": "my_torch_model",
                                                       "lstm_use_prev_action": True,
                                                       "lstm_use_prev_reward": False,
                                                       "custom_model_config": {
                                                       },
                                                   },
                                                   ) \
                .framework(framework="torch") \
                .environment(observation_space=_observation_space, action_space=_action_space,
                             disable_env_checking=True, ) \
                .rollouts(num_rollout_workers=0)
            self.algorithm = impala_config.build()
            self.algorithm.restore(checkpoint_path)
            self._policy = self.algorithm.workers.local_worker().get_policy()
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

    async def __call__(self, request: Request):
        compressed_input = await request.body()
        json_input = json_unzip(compressed_input)

        obs = [np.array(x) for x in json_input["observation"]]
        state1 = np.array([np.array(x) for x in json_input["state1"]])
        state2 = np.array([np.array(x) for x in json_input["state2"]])
        prev_a = json_input["prev_a"]
        prev_r = json_input["prev_r"]

        action, state_out, _ = self._policy.compute_actions(
            obs_batch=np.array(obs),
            state_batches=[state1, state2],
            prev_action_batch=prev_a,
            prev_reward_batch=prev_r)

        value = self._policy.model.value_function()
        normal = self._policy.model.normalized_value_function()
        compressed_output = json_zip({"action": action, "state_h": state_out[0], "state_c": state_out[1], "value": value.to('cpu').detach().numpy() })
        return compressed_output

if __name__ == "__main__":
    try:
        ray.init(address="auto", namespace="serve")
        serve.start(detached=True)
        impala_model = ServeModel.bind("D:\checkpoint\model_20240219")
        serve.run(impala_model)
    finally:
        ray.shutdown()