import ray

from impala_custom import ImpalaConfig
from ray import serve
from ray.rllib.models import ModelCatalog
from custom_model import CustomRNNModel
from starlette.requests import Request
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
import numpy as np
from gym.spaces import Discrete, Box
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
torch.backends.cudnn.benchmark = True
ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
_action_space = Discrete(3)
_observation_space = Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
@serve.deployment(route_prefix="/model", ray_actor_options={"num_gpus": 0.5})
class ServeModel:
    def __init__(self, checkpoint_path):
        try:
            impala_config = ImpalaConfig()
            impala_config = impala_config.training(gamma=0.99, lr=0.001, train_batch_size=500,
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
        json_input = await request.json()
        obs = [np.array(x) for x in json_input["observation"]]
        state1 = np.array([np.array(x) for x in json_input["state1"]])
        prev_a = json_input["prev_a"]
        prev_r = json_input["prev_r"]

        action, state_out, _ = self._policy.compute_actions(
            obs_batch=torch.from_numpy(np.array(obs).astype('float32')),
            state_batches=[torch.from_numpy(state1.astype('float32'))],
            prev_action_batch=prev_a,
            prev_reward_batch=prev_r)

        return {"action": action, "state_h": state_out[0] }

if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    serve.start(detached=True)
    ServeModel.deploy("D:\checkpoint\checkpoint_002169")