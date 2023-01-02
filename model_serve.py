import ray
from impala_custom import ImpalaCustomTrainer
from ray import serve
from ray.rllib.agents import impala
from ray.rllib.models import ModelCatalog
from custom_model import CustomRNNModel
from starlette.requests import Request
import numpy as np
from gym.spaces import Discrete, Box

ModelCatalog.register_custom_model("my_torch_model", CustomRNNModel)
_action_space = Discrete(3)
_observation_space = Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
@serve.deployment(route_prefix="/model", ray_actor_options={"num_gpus": 0.5})
class ServeModel:
    def __init__(self, checkpoint_path):
        try:
            config = impala.DEFAULT_CONFIG.copy()
            config.update({
                    "num_workers": 0,  # parallelism
                    "observation_space": _observation_space,
                    "action_space": _action_space,
                    "framework": "torch",
                    "model": {
                        "custom_model": "my_torch_model",
                        "lstm_use_prev_action": True,
                        "lstm_use_prev_reward": True,
                        "lstm_cell_size": 4,

                        # Extra kwargs to be passed to your model's c'tor.
                        "custom_model_config": {
                        },
                    },
                    "num_gpus": 1,
                    "log_level": 'DEBUG',
                })
            config['num_workers'] = 0
            self.trainer = ImpalaCustomTrainer(
                config=config)
            self.trainer.restore(checkpoint_path)
            self._policy = self.trainer.workers.local_worker().get_policy()
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

    async def __call__(self, request: Request):
        json_input = await request.json()
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

        return {"action": action, "state_h": state_out[0], "state_c": state_out[1] }

if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    serve.start(detached=True)
    ServeModel.deploy("D:\checkpoint\checkpoint_003762\checkpoint-3762")