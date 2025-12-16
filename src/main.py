import random
from itertools import count

import gymnasium as gym
import torch

from dqn import DQN
from replay import ReplayBuffer, Transition

device = "cpu"

env = gym.make("CarRacing-v3", continuous=False, domain_randomize=False)

random.seed(0)
torch.manual_seed(0)
state, info = env.reset(seed=0)
env.action_space.seed(0)
env.observation_space.seed(0)

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 100_000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 50_000
EPISODES = 3000
TARGET_UPDATE_SAMPLE_COUNT = 5000
LOSS_FN: torch.nn.MSELoss = torch.nn.MSELoss()


def get_epsilon(step):
    return (
        EPSILON_END
        + (EPSILON_START - EPSILON_END)
        * torch.exp(torch.tensor(-1.0 * step / EPSILON_DECAY)).item()
    )


episode_over = False

replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE)

q_network = DQN()
target_network = DQN()

optimizer = torch.optim.AdamW(q_network.parameters(), lr=LEARNING_RATE, amsgrad=True)

target_network.load_state_dict(q_network.state_dict())

replay_buffer.bootstrap(env)

observation = torch.Tensor(state, device=device).permute(2, 0, 1) / 255.0

episode_over = False
global_step = 0

for e in range(EPISODES):
    print(f"Episode: {e}")
    episode_reward = 0
    for t_c in count():
        if episode_over:
            observation, info = env.reset()
            observation = observation.permute(2, 0, 1, device=device) / 255.0
            break

        global_step += 1
        epsilon = get_epsilon(global_step)

        if t_c % TARGET_UPDATE_SAMPLE_COUNT == 0:
            target_network.load_state_dict(q_network.state_dict())

        action = q_network.epsilon_action(epsilon, observation, env.action_space)

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = torch.Tensor(next_observation).permute(2, 0, 1) / 255.0
        episode_reward += float(reward)
        episode_over = terminated or truncated

        t = Transition(
            observation, action, float(reward), next_observation, episode_over
        )

        replay_buffer.push(t)

        batch = replay_buffer.get_batch()

        observations = torch.stack([b.observation for b in batch])
        actions = torch.tensor([b.action for b in batch], dtype=torch.int)
        rewards = torch.Tensor([b.reward for b in batch])
        next_observations = torch.stack([b.next_observation for b in batch])
        episodes_over = torch.Tensor([b.done for b in batch])

        q_network.train()
        q_values = q_network.forward(observations)[torch.arange(32), actions]
        with torch.no_grad():
            target_q_values = rewards + (
                GAMMA
                * target_network.forward(next_observations).max(1).values
                * (1 - episodes_over)
            )
        loss = LOSS_FN(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q_network.parameters(), 10)
        optimizer.step()

        observation = next_observation

        print(
            f"Episode: {e} Loss: {loss} Reward: {episode_reward} Replay Buffer: {(float(len(replay_buffer)) / REPLAY_BUFFER_SIZE):.2f}%"
        )
