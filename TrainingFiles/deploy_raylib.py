import gymnasium as gym
import numpy
import cv2

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

import time

register_env('racing', lambda config: gym.make("CarRacing-v2", continuous = False))


config = (
    PPOConfig()
    .environment('racing')
    .rollouts(num_rollout_workers=5)
    .resources(num_cpus_per_worker=0.15,num_gpus=0)
    .framework("torch")
    .training(model={"fcnet_hiddens":[64,64]}))

algo = config.build()
algo.restore('./BigL/checkpoint_000801')

env = gym.make("CarRacing-v2", continuous=False, render_mode='human')
obs, _ = env.reset()

done = False
while not done:
    action = algo.compute_single_action(obs, explore=False)
    print(type(action))
    print(action)
    obs, reward, done, term, info = env.step(action)
    if done or term:
        break
print("Finished")