import sys
from pathlib import Path

from numpy.lib.format import open_memmap

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict
import torchvision.transforms as transforms
import gym
from rich.progress import track
from torchtyping import TensorType
from beartype import beartype
from beartype.typing import Iterator, Tuple, Union
from rich.progress import track
import cv2

# just force training on 64 bit systems

assert sys.maxsize > (
    2**32
), "you need to be on 64 bit system to store > 2GB experience for your q-transformer agent"

# constants

STATES_FILENAME = "states.memmap.npy"
ACTIONS_FILENAME = "actions.memmap.npy"
REWARDS_FILENAME = "rewards.memmap.npy"
DONES_FILENAME = "dones.memmap.npy"

DEFAULT_REPLAY_MEMORIES_FOLDER = "./value_function_memories_data"


# helpers
def exists(v):
    return v is not None


def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t


# replay memory dataset
class ReplayMemoryDataset(Dataset):
    @beartype
    def __init__(self, dataset_folder: str, num_timesteps: int = 1):
        assert num_timesteps >= 1, "num_timesteps must be at least 1"
        self.is_single_timestep = num_timesteps == 1
        self.num_timesteps = num_timesteps

        folder = Path(dataset_folder)
        assert (
            folder.exists() and folder.is_dir()
        ), "Folder must exist and be a directory"

        states_path = folder / STATES_FILENAME
        actions_path = folder / ACTIONS_FILENAME
        rewards_path = folder / REWARDS_FILENAME
        dones_path = folder / DONES_FILENAME

        self.states = open_memmap(str(states_path), dtype="float32", mode="r")
        self.actions = open_memmap(str(actions_path), dtype="int", mode="r")
        self.rewards = open_memmap(str(rewards_path), dtype="float32", mode="r")
        self.dones = open_memmap(str(dones_path), dtype="bool", mode="r")

        self.episode_length = (self.dones.cumsum(axis=-1) == 0).sum(axis=-1) + 1
        self.num_episodes, self.max_episode_len = self.dones.shape
        trainable_episode_indices = self.episode_length >= num_timesteps

        assert self.dones.size > 0, "no episodes found"

        self.num_episodes, self.max_episode_len = self.dones.shape

        timestep_arange = torch.arange(self.max_episode_len)

        timestep_indices = torch.stack(
            torch.meshgrid(torch.arange(self.num_episodes), timestep_arange), dim=-1
        )
        trainable_mask = timestep_arange < ((torch.from_numpy(self.episode_length) - num_timesteps).unsqueeze(1))
        self.indices = timestep_indices[trainable_mask]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        episode_index, timestep_index = self.indices[idx]
        timestep_slice = slice(timestep_index, (timestep_index + self.num_timesteps))
        states = self.states[episode_index, timestep_slice].copy()
        actions = self.actions[episode_index, timestep_slice].copy()
        rewards = self.rewards[episode_index, timestep_slice].copy()
        dones = self.dones[episode_index, timestep_slice].copy()
        next_state = self.states[
            episode_index, min(timestep_index, self.max_episode_len - 1)
        ].copy()
        return states, actions, rewards, dones, next_state


class SampleData:
    @beartype
    def __init__(
        self,
        env,
        memories_dataset_folder: str = DEFAULT_REPLAY_MEMORIES_FOLDER,
        num_episodes: int = 10,
        max_num_steps_per_episode: int = 13000,
    ):
        super().__init__()
        self.env = env
        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode

        mem_path = Path(memories_dataset_folder)
        self.memories_dataset_folder = mem_path

        mem_path.mkdir(exist_ok=True, parents=True)
        assert mem_path.is_dir()

        states_path = mem_path / STATES_FILENAME
        actions_path = mem_path / ACTIONS_FILENAME
        rewards_path = mem_path / REWARDS_FILENAME
        dones_path = mem_path / DONES_FILENAME

        prec_shape = (num_episodes, max_num_steps_per_episode)
        state_shape = (3, 32, 32)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.states = open_memmap(
            str(states_path),
            dtype="float32",
            mode="w+",
            shape=(*prec_shape, *state_shape),
        )

        self.actions = open_memmap(
            str(actions_path), dtype="int", mode="w+", shape=(*prec_shape, 1)
        )

        self.rewards = open_memmap(
            str(rewards_path), dtype="float32", mode="w+", shape=prec_shape
        )
        self.dones = open_memmap(
            str(dones_path), dtype="bool", mode="w+", shape=prec_shape
        )

    @beartype
    @torch.no_grad()
    def start_smple(self):
        for episode in range(self.num_episodes):
            print(f"episode {episode}")
            curr_state, log = self.env.reset()
            curr_state = self.transform(curr_state)
            for step in track(range(self.max_num_steps_per_episode)):
                last_step = step == (self.max_num_steps_per_episode - 1)

                action = self.env.action_space.sample()
                next_state, reward, termiuted, tuned, log = self.env.step(action)
                next_state = self.transform(next_state)
                done = termiuted | tuned | last_step
                # store memories using memmap, for later reflection and learning
                self.states[episode, step] = curr_state
                self.actions[episode, step] = action
                self.rewards[episode, step] = reward
                self.dones[episode, step] = done
                # if done, move onto next episode
                if done:
                    break
                # set next state
                curr_state = next_state

            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()

        del self.states
        del self.actions
        del self.rewards
        del self.dones
        self.memories_dataset_folder.resolve()
        print(f"completed")
