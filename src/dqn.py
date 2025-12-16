import random

import torch
from gymnasium import Space
from torch import nn
from torch.types import Number


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Input (B, 3, 96, 96)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Input (B, 64, 48, 48)
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Input (B, 32, 24, 24)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Input (B, 16, 12, 12)
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (B, 8, 6, 6)
            nn.Flatten(
                start_dim=1,
            ),
            # Input (B, 288)
            nn.Linear(8 * 6 * 6, 5),
            # Output (B, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def epsilon_action(
        self, epsilon: float, state: torch.Tensor, action_space: Space
    ) -> Number:
        r = random.random()
        if r < epsilon:
            return action_space.sample()

        with torch.no_grad():
            return self.forward(state.reshape((1, 3, 96, 96))).argmax().item()
