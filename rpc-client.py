import multiprocessing
import warnings
warnings.simplefilter("ignore", UserWarning)
SERVER_BASE_PORT = 9900
env_cfg = {
    "NUM_STATES": 8,
    "NUM_ACTIONS": 3,

    "FEE": 0.1,
    "MAX_EP": 12000,
    "DF_SIZE": 186982,
    "frameskip": 3,
    "mode": "train",
}
ACTORS = 16

class Agent(multiprocessing.Process):
    stop_signal = False

    def __init__(self, worker_idx = 0):
        super(Agent, self).__init__()
        self.worker_idx = worker_idx

    def run(self):
        import warnings
        warnings.simplefilter("ignore", UserWarning)
        from crypto_env import CryptoEnv
        from ray.rllib.env.policy_client import PolicyClient
        self.stop_signal = False
        env = CryptoEnv(config=env_cfg, worker_idx=self.worker_idx)

        client = PolicyClient(
            f"http://localhost:{SERVER_BASE_PORT}", inference_mode="remote"
        )
        obs, info = env.reset()
        eid = client.start_episode(training_enabled=True)
        rewards = 0.0
        i = 0
        while not self.stop_signal:
            action = client.get_action(eid, obs)
            # Perform a step in the external simulator (env).
            obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward

            # Log next-obs, rewards, and infos.
            client.log_returns(eid, reward, info=info)
            # Reset the episode if done.
            if terminated or truncated:
                rewards = 0.0

                # End the old episode.
                client.end_episode(eid, obs)

                # Start a new episode.
                obs, info = env.reset()
                eid = client.start_episode(training_enabled=True)

if __name__ == '__main__':
    envs = [Agent(worker_idx = i) for i in range(ACTORS)]
    for e in envs:
        e.daemon = True
        e.start()
    print("running")
    [proc.join() for proc in envs]