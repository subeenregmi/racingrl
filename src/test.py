import random
import time
from math import trunc

import gymnasium as gym
import torch

from dqn import DQN

env = gym.make(
    "CarRacing-v3", render_mode="human", continuous=False, domain_randomize=False
)

random.seed(0)
torch.manual_seed(0)
state, info = env.reset(seed=0)
env.action_space.seed(0)
env.observation_space.seed(0)

q_network = DQN()
q_network.load_state_dict(torch.load("model.pth", map_location="cpu"))

episode_over = False

observation, info = env.reset()

observation = torch.Tensor(observation).permute(2, 0, 1) / 255.0

epsilon = 0.0

while not episode_over:
    action = q_network.epsilon_action(epsilon, observation, env.action_space)
    observation, reward, terminated, truncated, info = env.step(action)
    observation = torch.Tensor(observation).permute(2, 0, 1) / 255.0

    env.render()

    episode_over = terminated or truncated
