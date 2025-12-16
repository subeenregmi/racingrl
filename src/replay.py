import random
from collections import deque, namedtuple

import gymnasium
import torch

Transition = namedtuple(
    "Transition", ("observation", "action", "reward", "next_observation", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int):
        self.store = deque["Transition"]([], maxlen=capacity)
        self.batch_size = batch_size

    def get_batch(self):
        return random.sample(self.store, self.batch_size)

    def __len__(self):
        return len(self.store)

    def push(self, t: Transition):
        self.store.append(t)

    def bootstrap(self, env: gymnasium.Env):
        state, _ = env.reset()

        observation = torch.Tensor(state).permute(2, 0, 1) / 255.0
        for _ in range(self.batch_size):
            random_action = env.action_space.sample()

            next_observation, reward, terminated, truncated, _ = env.step(random_action)
            next_observation = torch.Tensor(next_observation).permute(2, 0, 1) / 255.0

            t = Transition(
                observation,
                random_action,
                reward,
                next_observation,
                terminated or truncated,
            )

            self.push(t)

            if terminated or truncated:
                observation, _ = env.reset()
                continue

            observation = next_observation

        env.reset()
