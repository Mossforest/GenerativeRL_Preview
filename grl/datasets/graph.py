from abc import abstractmethod
from typing import List

import gym
import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data import LazyTensorStorage, LazyMemmapStorage
from torch_geometric.data import InMemoryDataset, Data
from grl.utils.log import log
from grl.datasets.gp import GPD4RLDataset



class D4RLGraphDataset(InMemoryDataset):
    """
    Overview:
        D4RL Dataset for Generative Policy algorithm.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``.
    """

    def __init__(
        self,
        env_id: str,
        action_augment_num: int = None,
    ):
        """
        Overview:
            Initialization method of GPD4RLDataset class
        Arguments:
            env_id (:obj:`str`): The environment id
        """

        super().__init__()
        import d4rl

        data = d4rl.qlearning_dataset(gym.make(env_id))
        self.states = torch.from_numpy(data["observations"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.next_states = torch.from_numpy(data["next_observations"]).float()
        reward = torch.from_numpy(data["rewards"]).view(-1, 1).float()
        self.is_finished = torch.from_numpy(data["terminals"]).view(-1, 1).float()

        reward_tune = "iql_antmaze" if "antmaze" in env_id else "iql_locomotion"
        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            min_ret, max_ret = GPD4RLDataset.return_range(data, 1000)
            reward /= max_ret - min_ret
            reward *= 1000
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        self.len = self.states.shape[0]
        log.info(f"{self.len} data loaded in GPD4RLDataset")
        # self.storage.set(
        #     range(self.len), TensorDict(
        #         {
        #             "s": self.states,
        #             "a": self.actions,
        #             "r": self.rewards,
        #             "s_": self.next_states,
        #             "d": self.is_finished,
        #         },
        #         batch_size=[self.len],
        #     )
        # )
        
        self.graph_data = []
        
        def _build_graph(self, s, a, r, s_, d):
            
            data = Data(x=s,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=s_ #you can add more arguments as you like
                        )
            return data

    def return_range(dataset, max_episode_steps):
        returns, lengths = [], []
        ep_ret, ep_len = 0.0, 0
        for r, d in zip(dataset["rewards"], dataset["terminals"]):
            ep_ret += float(r)
            ep_len += 1
            if d or ep_len == max_episode_steps:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0.0, 0
        # returns.append(ep_ret)    # incomplete trajectory
        lengths.append(ep_len)  # but still keep track of number of steps
        assert sum(lengths) == len(dataset["rewards"])
        return min(returns), max(returns)


    def __getitem__(self, index):
        """
        Overview:
            Get data by index
        Arguments:
            index (:obj:`int`): Index of data
        Returns:
            data (:obj:`dict`): Data dict
        
        .. note::
            The data dict contains the following keys:
            
            s (:obj:`torch.Tensor`): State
            a (:obj:`torch.Tensor`): Action
            r (:obj:`torch.Tensor`): Reward
            s_ (:obj:`torch.Tensor`): Next state
            d (:obj:`torch.Tensor`): Is finished
            fake_a (:obj:`torch.Tensor`): Fake action for contrastive energy prediction and qgpo training \
                (fake action is sampled from the action support generated by the behaviour policy)
            fake_a_ (:obj:`torch.Tensor`): Fake next action for contrastive energy prediction and qgpo training \
                (fake action is sampled from the action support generated by the behaviour policy)
        """

        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.len
    
    @property
    def raw_file_name(self):
        pass
    
    @property
    def processed_file_names(self):
        pass
    
    def download(self):
        pass
    
    def process(self):
        pass
