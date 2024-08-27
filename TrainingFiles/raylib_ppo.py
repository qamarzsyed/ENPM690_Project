import gymnasium as gym

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

import time

register_env('racing', lambda config: gym.make("CarRacing-v2", continuous = False))


config = (
    PPOConfig()
    .environment('racing')
    .rollouts(num_rollout_workers=30)
    .resources(num_cpus_per_worker=0.08,num_gpus=0)
    .framework("torch")
    .training(model={"fcnet_hiddens":[64,64]}))


algo = config.build()
for i in range(5000):
    algo.train()
    if i % 100 == 0:
        algo.save("./BigL/")