from ray.rllib.algorithms.impala import ImpalaConfig
config = ImpalaConfig()
config = config.training(lr=0.0003, train_batch_size=500)
config = config.resources(num_gpus=1)
config = config.rollouts(num_rollout_workers=1)
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="CartPole-v1")
algo.train()
del algo