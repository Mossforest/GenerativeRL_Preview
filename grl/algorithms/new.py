import copy
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from rich.progress import Progress, track
from tensordict import TensorDict
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions.transforms import TanhTransform
from torch.utils.data import DataLoader

import wandb
from grl.generative_models.diffusion_model.energy_conditional_diffusion_model import \
    EnergyConditionalDiffusionModel
from grl.neural_network import (MultiLayerPerceptron, get_module,
                                register_module)
from grl.neural_network.encoders import (
    ExponentialFourierProjectionTimeEncoder, get_encoder)
from grl.neural_network.transformers.dit import (DiTBlock, FinalLayer1D,
                                                 get_1d_pos_embed, modulate)
from grl.rl_modules.simulators import create_simulator
from grl.rl_modules.value_network import DoubleQNetwork, DoubleVNetwork
from grl.utils.config import merge_two_dicts_into_newone
from grl.utils.log import log


class DiffusionModelDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for training the diffusion model.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``.
    """

    def __init__(
            self,
            t_length: int,
            data: List = None,
            device: str = None,
        ):
        """
        Overview:
            Initialization.
        Arguments:
            t_length (:obj:`int`): The length of the time sequence for the diffusion model.
            data (:obj:`List`): The data list
            device (:obj:`str`): The device to store the dataset
        """

        super().__init__()
        self.device = "cpu" if device is None else device

        if data is not None:
            self.state = torch.from_numpy(data['state']).float().to(device)
            self.next_states = torch.from_numpy(data['next_states']).float().to(device)
            self.len = self.states.shape[0]
        else:
            self.state = torch.tensor([]).to(device)
            self.next_states = torch.tensor([]).to(device)
            self.len = 0
        log.debug(f"{self.len} data loaded in DiffusionModelDataset")

        self.cache = []

    def drop_data(self, drop_ratio: float, random: bool = True):
        # drop the data from the dataset
        drop_num = int(self.len * drop_ratio)
        # randomly drop the data if random is True
        if random:
            drop_indices = torch.randperm(self.len)[:drop_num].to(self.device)
        else:
            drop_indices = torch.arange(drop_num).to(self.device)
        keep_mask = torch.ones(self.len, dtype=torch.bool).to(self.device)
        keep_mask[drop_indices] = False
        self.state = self.states[keep_mask]
        self.next_states = self.next_states[keep_mask]
        self.len = self.states.shape[0]
        log.debug(f"{drop_num} data dropped in DiffusionModelDataset")
        
    def load_data(self, data: List):
        # concatenate the data into the dataset
        device = self.device
        # collate the data by sorting the keys

        # keys = ["obs", "action", "done", "next_obs", "reward"]
        collated_data = data

        # collated_data['obs'].shape: (N, D)
        # collated_data['next_obs'].shape: (N, D)
        # collated_data['action'].shape: (N, A)
        # collated_data['reward'].shape: (N, 1)
        # collated_data['done'].shape: (N, 1)

        # collated_data['done'] is an array containing False or True, get index of True
        done_indices = np.where(collated_data['done'])[0]
            
        self.states = torch.cat([self.states, collated_data['obs'].float()], dim=0)
        self.actions = torch.cat([self.actions, collated_data['action'].float()], dim=0)
        self.next_states = torch.cat([self.next_states, collated_data['next_obs'].float()], dim=0)
        reward = collated_data['reward'].view(-1, 1).float()
        self.is_finished = torch.cat([self.is_finished, collated_data['done'].view(-1, 1).float()], dim=0)
        self.rewards = torch.cat([self.rewards, reward], dim=0)
        self.len = self.states.shape[0]
        log.debug(f"{self.len} data loaded in Dataset")

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

        data = {
            's': self.states[index % self.len],
            'a': self.actions[index % self.len],
            'r': self.rewards[index % self.len],
            's_': self.next_states[index % self.len],
            'd': self.is_finished[index % self.len],
            'fake_a': self.fake_actions[index % self.len]
            if hasattr(self, "fake_actions") else 0.0,  # self.fake_actions <D, 16, A>
            'fake_a_': self.fake_next_actions[index % self.len]
            if hasattr(self, "fake_next_actions") else 0.0,  # self.fake_next_actions <D, 16, A>
        }
        return data

    def __len__(self):
        return self.len


class Dataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for QGPO algorithm. The training of QGPO algorithm is based on contrastive energy prediction, \
        which needs true action and fake action. The true action is sampled from the dataset, and the fake action \
        is sampled from the action support generated by the behaviour policy.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``.
    """

    def __init__(
            self,
            data: List = None,
            device: str = None,
        ):
        """
        Overview:
            Initialization method.
        Arguments:
            data (:obj:`List`): The data list
            device (:obj:`str`): The device to store the dataset
        """

        super().__init__()
        self.device = "cpu" if device is None else device

        if data is not None:
            self.states = torch.from_numpy(data['observations']).float().to(device)
            self.actions = torch.from_numpy(data['actions']).float().to(device)
            self.next_states = torch.from_numpy(data['next_observations']).float().to(device)
            reward = torch.from_numpy(data['rewards']).view(-1, 1).float().to(device)
            self.is_finished = torch.from_numpy(data['terminals']).view(-1, 1).float().to(device)
            self.rewards = reward
            self.len = self.states.shape[0]
        else:
            self.states = torch.tensor([]).to(device)
            self.actions = torch.tensor([]).to(device)
            self.next_states = torch.tensor([]).to(device)
            self.is_finished = torch.tensor([]).to(device)
            self.rewards = torch.tensor([]).to(device)
            self.len = 0
        log.debug(f"{self.len} data loaded in Dataset")
        self.diffusion_model_dataset = DiffusionModelDataset(t_length=100, data=data, device=device)


    def drop_data(self, drop_ratio: float, random: bool = True):
        # drop the data from the dataset
        drop_num = int(self.len * drop_ratio)
        # randomly drop the data if random is True
        if random:
            drop_indices = torch.randperm(self.len)[:drop_num].to(self.device)
        else:
            drop_indices = torch.arange(drop_num).to(self.device)
        keep_mask = torch.ones(self.len, dtype=torch.bool).to(self.device)
        keep_mask[drop_indices] = False
        self.states = self.states[keep_mask]
        self.actions = self.actions[keep_mask]
        self.next_states = self.next_states[keep_mask]
        self.is_finished = self.is_finished[keep_mask]
        self.rewards = self.rewards[keep_mask]
        self.len = self.states.shape[0]
        log.debug(f"{drop_num} data dropped in Dataset")
        
    def load_data(self, data: List):
        # concatenate the data into the dataset
        device = self.device
        # collate the data by sorting the keys

        keys = ["obs", "action", "done", "next_obs", "reward"]

        collated_data = {
            k: torch.tensor(np.stack([item[k] for item in data])).to(device)
            for i, k in enumerate(keys)
        }

        self.states = torch.cat([self.states, collated_data['obs'].float()], dim=0)
        self.actions = torch.cat([self.actions, collated_data['action'].float()], dim=0)
        self.next_states = torch.cat([self.next_states, collated_data['next_obs'].float()], dim=0)
        reward = collated_data['reward'].view(-1, 1).float()
        self.is_finished = torch.cat([self.is_finished, collated_data['done'].view(-1, 1).float()], dim=0)
        self.rewards = torch.cat([self.rewards, reward], dim=0)
        self.len = self.states.shape[0]
        log.debug(f"{self.len} data loaded in Dataset")

        self.diffusion_model_dataset.load_data(collated_data)


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

        data = {
            's': self.states[index % self.len],
            'a': self.actions[index % self.len],
            'r': self.rewards[index % self.len],
            's_': self.next_states[index % self.len],
            'd': self.is_finished[index % self.len],
            'fake_a': self.fake_actions[index % self.len]
            if hasattr(self, "fake_actions") else 0.0,  # self.fake_actions <D, 16, A>
            'fake_a_': self.fake_next_actions[index % self.len]
            if hasattr(self, "fake_next_actions") else 0.0,  # self.fake_next_actions <D, 16, A>
        }
        return data

    def __len__(self):
        return self.len


class StateCritic(nn.Module):
    """
    Overview:
        Critic network of New algorithm for state.

        .. math::
            V(s_t)

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of New critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.v = DoubleVNetwork(config.DoubleVNetwork)
        self.v_target = copy.deepcopy(self.v).requires_grad_(False)

    def forward(
            self,
            state: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the output of New critic.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
        """

        return self.v(state)

    def compute_double_v(
            self,
            state: Union[torch.Tensor, TensorDict] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Return the output of two v networks.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            v1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first V network.
            v2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second V network.
        """
        return self.v.compute_double_v(state)

    def v_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the v loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        """
        with torch.no_grad():
            next_v = self.v_target(next_state)
        # Update V function
        targets = reward + (1. - done.float()) * discount_factor * next_v
        v0, v1 = self.v.compute_double_v(state)
        v_loss = (torch.nn.functional.mse_loss(v0, targets) + torch.nn.functional.mse_loss(v1, targets)) / 2
        return v_loss

class StateSequenceCritic(nn.Module):
    """
    Overview:
        Critic network of New algorithm for state sequence.

        .. math::
            V(s_{t:t+T})

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of New critic network.
        Arguments:
            config (:obj:`EasyDict`): The configuration dict.
        """

        super().__init__()
        self.config = config
        self.state_critic = StateCritic(config.state_critic)
        self.v = DoubleVNetwork(config.DoubleVNetwork)
        self.v_target = copy.deepcopy(self.v).requires_grad_(False)

    def forward(
            self,
            states: Union[torch.Tensor, TensorDict] = None,
        ) -> torch.Tensor:
        """
        Overview:
            Return the output of New critic.
        Arguments:
            states (:obj:`torch.Tensor`): The input states.
        """

        return self.v(states)

    def compute_double_v(
            self,
            states: Union[torch.Tensor, TensorDict] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Return the output of two v networks.
        Arguments:
            states (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            v1 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the first V network.
            v2 (:obj:`Union[torch.Tensor, TensorDict]`): The output of the second V network.
        """
        return self.v.compute_double_v(states)

    def state_critic_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the state critic loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            discount_factor (:obj:`float`): The discount factor.
        """

        return self.state_critic.v_loss(state, reward, next_state, done, discount_factor)

    def v_loss(
            self,
            states: Union[torch.Tensor, TensorDict],
            rewards: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the v loss.
        Arguments:
            states (:obj:`torch.Tensor`): The input states.
            rewards (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        Returns:
            v_loss (:obj:`torch.Tensor`): The V loss.
        Shape:
            - states: :math:`(N, T, D)`, where :math:`N` is the batch size, :math:`T` is the time step, :math:`D` is the state dimension.
            - rewards: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - next_state: :math:`(N, D)`, where :math:`N` is the batch size, :math:`D` is the state dimension.
            - done: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - discount_factor: :math:`()`.
            - v_loss: :math:`()`.
        """
        with torch.no_grad():
            states_1_T_1 = torch.concat([states[:,1:], next_state.unsqueeze(1)], dim=1)
            # state_1_T_1_shape: (N, T, D)
            states_1_T_1_reshape = states_1_T_1.reshape(states_1_T_1.shape[0] * states_1_T_1.shape[1], *states_1_T_1.shape[2:])
            # states_1_T_1_reshape_shape: (N * T, D)
            next_v_reshape = self.state_critic(states_1_T_1_reshape)
            # next_v_reshape_shape: (N * T, 1)
            next_v = next_v_reshape.reshape(states_1_T_1.shape[0], states_1_T_1.shape[1])
            # next_v_shape: (N, T)

            
            # rewards_shape: (N, T)
            # get up_triangle, shape: (T, T)
            up_traingle = torch.triu(torch.ones(rewards.shape[1], rewards.shape[1]), diagonal=1)
            # sum_rewards = torch.einsum("ij,nj->nj", up_traingle, rewards)
            sum_rewards = torch.einsum("ij,nj->ni", up_traingle, rewards)
            # sum_rewards_shape: (N, T)
            # done_shape: (N, 1)
            done_reshape = done.unsqueeze(1).repeat(1, rewards.shape[1])
            # done_reshape_shape: (N, T)

        # Update V function
        targets = sum_rewards + (1. - done.float()) * discount_factor * next_v
        # targets_shape: (N, T)
        targets_mean = targets.mean(dim=1, keepdim=True)
        # targets_mean_shape: (N, 1)

        v0, v1 = self.v.compute_double_v(states)
        # v0_shape: (N, 1)
        # v1_shape: (N, 1)

        v_loss = (torch.nn.functional.mse_loss(v0, targets_mean) + torch.nn.functional.mse_loss(v1, targets_mean)) / 2
        return v_loss

class NonegativeFunction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = MultiLayerPerceptron(**config)

    def forward(self, x):
        return torch.exp(self.model(x))

class TanhFunction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = TanhTransform(cache_size=1)
        self.model = MultiLayerPerceptron(**config)

    def forward(self, x):
        return self.transform(self.model(x))
    
class CovarianceMatrix(nn.Module):

    def __init__(self, config, delta=1e-8):
        super().__init__()
        self.dim = config.dim

        self.sigma_lambda = NonegativeFunction(config.sigma_lambda)
        self.sigma_offdiag = TanhFunction(config.sigma_offdiag)

        # register eye matrix
        self.eye = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        self.delta = delta
        
    def low_triangle_matrix(self, x):
        low_t_m = self.eye.detach()

        low_t_m=low_t_m.repeat(x.shape[0],1,1)
        # low_t_m[torch.concat((torch.reshape(torch.arange(x.shape[0]).repeat(self.dim*(self.dim-1)//2,1).T,(1,-1)),torch.tril_indices(self.dim, self.dim, offset=-1).repeat(1,x.shape[0]))).tolist()]=torch.reshape(self.sigma_offdiag(x),(-1,1)).squeeze(-1)
        low_t_m = low_t_m + torch.triu(self.sigma_offdiag(x), diagonal=1)
        lambda_ = self.delta + self.sigma_lambda(x)
        low_t_m=torch.einsum("bj,bjk,bk->bjk", lambda_, low_t_m, lambda_)

        return low_t_m

    def forward(self,x):
        ltm = self.low_triangle_matrix(x)
        return torch.matmul(ltm, ltm.T)

class Gaussian(nn.Module, Distribution):
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        if not hasattr(config, "condition_encoder"):
            self.condition_encoder = torch.nn.Identity()
        else:
            self.condition_encoder = get_encoder(config.condition_encoder.type)(**config.condition_encoder.args)
        self.mu_model = MultiLayerPerceptron(**config.mu_model)
        self.cov = CovarianceMatrix(config.cov)

    def dist(self, condition):
        mu=self.mu_model(condition)
        # repeat the sigma to match the shape of mu
        scale_tril = self.cov.low_triangle_matrix(condition)
        return MultivariateNormal(loc=mu, scale_tril = scale_tril)

    def log_prob(self, x, condition):
        return self.dist(condition).log_prob(x)

    def sample(self, condition, sample_shape=torch.Size()):
        return self.dist(condition).sample(sample_shape=sample_shape)
            
    def rsample(self, condition, sample_shape=torch.Size()): 
        return self.dist(condition).rsample(sample_shape=sample_shape)

    def entropy(self, condition):
        return self.dist(condition).entropy()

    def rsample_and_log_prob(self, condition, sample_shape=torch.Size()):
        dist=self.dist(condition)
        x=dist.rsample(sample_shape=sample_shape)
        log_prob=dist.log_prob(x)
        return x, log_prob

    def sample_and_log_prob(self, condition, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample_and_log_prob(condition, sample_shape)

    def forward(self, condition):
        dist=self.dist(condition)
        x=dist.rsample()
        log_prob=dist.log_prob(x)
        return x, log_prob

class GaussianPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gaussian(config.model)

        
    def forward(self, obs):
        action, logp = self.model(obs)
        return action, logp
    
    def log_prob(self, action, obs):
        return self.model.log_prob(action, obs)
    
    def sample(self, obs, sample_shape=torch.Size()):
        return self.model.sample(obs, sample_shape)
    
    def rsample(self, obs, sample_shape=torch.Size()):
        return self.model.rsample(obs, sample_shape)
    
    def entropy(self, obs):
        return self.model.entropy(obs)
    
    def dist(self, obs):
        return self.model.dist(obs)

class EnergyModel(nn.Module):
    
        def __init__(self, config: EasyDict, model: nn.Module):
            super().__init__()
            self.config = config
            self.model = model
    
        def forward(self, x, condition):
            # x.shape: (N, T, D)
            # condition.shape: (N, D)
            # transfer condition to (N, 1, D)
            condition = condition.unsqueeze(1)
            # condition.shape: (N, 1, D)
            # concat condition to x
            x = torch.cat([condition, x], dim=1)
            # x.shape: (N, T + 1, D)
            return self.energy(x)

class DiT1D_Special(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for 1D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        token_size: int,
        in_channels: int = 4,
        out_channels: int = None,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        config: EasyDict = None,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            in_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of input channels, defaults to 4.
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 1152.
            depth (:obj:`int`): The depth of transformer, defaults to 28.
            num_heads (:obj:`int`): The number of attention heads, defaults to 16.
            mlp_ratio (:obj:`float`): The hidden size of the MLP with respect to the hidden size of Attention, defaults to 4.0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.num_heads = num_heads

        self.x_embedder = nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=1, groups=in_channels, bias=False)
        if config and hasattr(config, "y_embedder"):
            self.y_embedder = get_module(config.y_embedder.type)(**config.y_embedder.args)
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_1d_pos_embed(embed_dim=hidden_size, grid_num=token_size)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float(), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        ):
        """
        Overview:
            Forward pass of DiT for 3D data.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of inputs with spatial information (originally at t=0 it is tensor of videos or latent representations of videos).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        # x is of shape (N, T, C), reshape to (N, C, T)
        x = torch.einsum('ntc->nct', x)
        # condition shape: (N, C)
        # concat condition to x to make shape (N, C, T+1) 
        x = torch.cat([condition.unsqueeze(-1), x], dim=-1)
        x = self.x_embedder(x) + torch.einsum("th->ht", self.pos_embed)
        t = self.t_embedder(t)                   # (N, hidden_size)
        c = t
        for block in self.blocks:
            x = block(x, c)                      # (N, T+1, hidden_size)
        x = self.final_layer(x, c)                # (N, T+1, C)
        x = x[:, 1:]                              # (N, T, C)
        return x

register_module(DiT1D_Special, "DiT1D_Special")

class ActionStateCritic(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.v = StateSequenceCritic(config.state_sequence_critic)
        self.energy_model = EnergyModel(config.energy_model, self.v)
        self.diffusion_model = EnergyConditionalDiffusionModel(config.diffusion_model, energy_model=self.energy_model)
        self.q = DoubleQNetwork(config.DoubleQNetwork)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)

    def compute_mininum_q(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        return self.q.compute_mininum_q(action, state)


    def forward(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        return self.q(action, state)

    def diffusion_model_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            next_states: Union[torch.Tensor, TensorDict],
        ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            next_states (:obj:`torch.Tensor`): The input next states.
        Shape:
            - state: :math:`(N, D)`, where :math:`N` is the batch size, :math:`D` is the state dimension.
            - next_states: :math:`(N, T, D)`, where :math:`N` is the batch size, :math:`T` is the time step, :math:`D` is the state dimension.
        """

        return self.diffusion_model.score_matching_loss(next_states, condition=state)

    def state_critic_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the state critic loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            discount_factor (:obj:`float`): The discount factor.
        """

        self.v.state_critic_loss(state, reward, next_state, done, discount_factor)

    def state_sequence_critic_loss(
            self,
            states: Union[torch.Tensor, TensorDict],
            rewards: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the v loss.
        Arguments:
            states (:obj:`torch.Tensor`): The input states.
            rewards (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        Returns:
            v_loss (:obj:`torch.Tensor`): The V loss.
        Shape:
            - states: :math:`(N, T, D)`, where :math:`N` is the batch size, :math:`T` is the time step, :math:`D` is the state dimension.
            - rewards: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - next_state: :math:`(N, D)`, where :math:`N` is the batch size, :math:`D` is the state dimension.
            - done: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - discount_factor: :math:`()`.
            - v_loss: :math:`()`.
        """

        self.v.v_loss(states, rewards, next_state, done, discount_factor)

    def energy_guidance_loss(
            self,
            fake_next_states: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the energy guidance loss of the diffusion model.
        Arguments:
            fake_next_states (:obj:`torch.Tensor`): The input fake next action.
            state (:obj:`torch.Tensor`): The input state.
        """

        self.diffusion_model.energy_guidance_loss(x=fake_next_states, condition=state)

    def q_loss(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_states: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        with torch.no_grad():
            # next_states.shape: (N, T, D)
            next_v = self.v(next_states)
            # next_v.shape: (N, 1)
        # Update Q function
        targets = reward + (1. - done.float()) * discount_factor * next_v
        q0, q1 = self.q.compute_double_q(action, state)
        q_loss = (torch.nn.functional.mse_loss(q0, targets) + torch.nn.functional.mse_loss(q1, targets)) / 2
        return q_loss

class Policy(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.device = config.device
        self.policy = GaussianPolicy(config.policy)
        self.critic = ActionStateCritic(config.critic)

    def forward(self, state: Union[torch.Tensor, TensorDict]) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of New policy, which is the action and log probability of action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.policy(state)
    
    def sample(
            self,
            state: Union[torch.Tensor, TensorDict],
        ) -> Union[torch.Tensor, TensorDict]:
        """
        Overview:
            Return the output of New policy, which is the action conditioned on the state.
        Arguments:
            state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            action (:obj:`Union[torch.Tensor, TensorDict]`): The output action.
        """
        return self.policy.sample(state)
    
    def diffusion_model_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            next_states: Union[torch.Tensor, TensorDict],
        ):
        """
        Overview:
            Calculate the behaviour policy loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            next_states (:obj:`torch.Tensor`): The input next states.
        Shape:
            - state: :math:`(N, D)`, where :math:`N` is the batch size, :math:`D` is the state dimension.
            - next_states: :math:`(N, T, D)`, where :math:`N` is the batch size, :math:`T` is the time step, :math:`D` is the state dimension.
        """

        return self.critic.diffusion_model_loss(state, next_states)
    
    def state_critic_loss(
            self,
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the state critic loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            reward (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            discount_factor (:obj:`float`): The discount factor.
        """

        return self.critic.state_critic_loss(state, reward, next_state, done, discount_factor)

    def state_sequence_critic_loss(
            self,
            states: Union[torch.Tensor, TensorDict],
            rewards: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the v loss.
        Arguments:
            states (:obj:`torch.Tensor`): The input states.
            rewards (:obj:`torch.Tensor`): The input reward.
            next_state (:obj:`torch.Tensor`): The input next state.
            done (:obj:`torch.Tensor`): The input done.
            fake_next_action (:obj:`torch.Tensor`): The input fake next action.
            discount_factor (:obj:`float`): The discount factor.
        Returns:
            v_loss (:obj:`torch.Tensor`): The V loss.
        Shape:
            - states: :math:`(N, T, D)`, where :math:`N` is the batch size, :math:`T` is the time step, :math:`D` is the state dimension.
            - rewards: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - next_state: :math:`(N, D)`, where :math:`N` is the batch size, :math:`D` is the state dimension.
            - done: :math:`(N, T)`, where :math:`N` is the batch size, :math:`T` is the time step.
            - discount_factor: :math:`()`.
            - v_loss: :math:`()`.
        """

        return self.critic.state_sequence_critic_loss(states, rewards, next_state, done, discount_factor)

    def energy_guidance_loss(
            self,
            fake_next_states: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
        ) -> torch.Tensor:
        """
        Overview:
            Calculate the energy guidance loss of the diffusion model.
        Arguments:
            fake_next_states (:obj:`torch.Tensor`): The input fake next action.
            state (:obj:`torch.Tensor`): The input state.
        """
    
        return self.critic.energy_guidance_loss(fake_next_states, state)

    def action_state_critic_loss(
            self,
            action: Union[torch.Tensor, TensorDict],
            state: Union[torch.Tensor, TensorDict],
            reward: Union[torch.Tensor, TensorDict],
            next_state: Union[torch.Tensor, TensorDict],
            done: Union[torch.Tensor, TensorDict],
            discount_factor: float = 1.0,
        ) -> torch.Tensor:

        return self.critic.q_loss(action, state, reward, next_state, done, discount_factor)

    def policy_loss(
            self,
            state: Union[torch.Tensor, TensorDict]
    ) -> torch.Tensor:
        """
        Overview:
            Calculate the policy loss.
        Arguments:
            state (:obj:`torch.Tensor`): The input state.
            action (:obj:`torch.Tensor`): The input action.
        """

        action, logp = self.policy(state)
        q_value = self.critic.compute_mininum_q(action, state)
        policy_loss = (self.entropy_coeffi.data * logp - q_value)
        return policy_loss, logp


class NewAlgorithm:

    def __init__(
        self,
        config:EasyDict = None,
        simulator = None,
        model: Union[torch.nn.Module, torch.nn.ModuleDict] = None,
    ):
        """
        Overview:
            Initialize the New algorithm.
        Arguments:
            config (:obj:`EasyDict`): The configuration , which must contain the following keys:
                train (:obj:`EasyDict`): The training configuration.
                deploy (:obj:`EasyDict`): The deployment configuration.
            simulator (:obj:`object`): The environment simulator.
            model (:obj:`Union[torch.nn.Module, torch.nn.ModuleDict]`): The model.
        Interface:
            ``__init__``, ``train``, ``deploy``
        """
        self.config = config
        self.simulator = simulator

        #---------------------------------------
        # Customized model initialization code ↓
        #---------------------------------------

        self.model = model if model is not None else torch.nn.ModuleDict()

        #---------------------------------------
        # Customized model initialization code ↑
        #---------------------------------------

    def train(
        self,
        config: EasyDict = None
    ):
        """
        Overview:
            Train the model using the given configuration. \
            A weight-and-bias run will be created automatically when this function is called.
        Arguments:
            config (:obj:`EasyDict`): The training configuration.
        """
        
        config = merge_two_dicts_into_newone(
            self.config.train if hasattr(self.config, "train") else EasyDict(),
            config
        ) if config is not None else self.config.train

        with wandb.init(
            project=config.project if hasattr(config, "project") else __class__.__name__,
            **config.wandb if hasattr(config, "wandb") else {}
        ) as wandb_run:
            config=merge_two_dicts_into_newone(EasyDict(wandb_run.config), config)
            wandb_run.config.update(config)
            self.config.train = config

            self.simulator = create_simulator(config.simulator) if hasattr(config, "simulator") else self.simulator
            self.dataset = Dataset(**config.dataset)

            #---------------------------------------
            # Customized model initialization code ↓
            #---------------------------------------

            if hasattr(config.model, "Policy"):
                self.model["Policy"] = Policy(config.model.Policy)
                self.model["Policy"].to(config.model.Policy.device)
                if torch.__version__ >= "2.0.0":
                    self.model["Policy"] = torch.compile(self.model["Policy"])

            #---------------------------------------
            # Customized model initialization code ↑
            #---------------------------------------


            #---------------------------------------
            # Customized training code ↓
            #---------------------------------------


            diffusion_model_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.diffusion_model.model.parameters(),
                lr=config.parameter.diffusion_model.learning_rate,
            )

            state_critic_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.v.state_critic.v.parameters(),
                lr=config.parameter.state_critic.learning_rate,
            )

            state_sequence_critic_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.v.v.parameters(),
                lr=config.parameter.state_sequence_critic.learning_rate,
            )

            energy_guidance_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.diffusion_model.energy_guidance.parameters(),
                lr=config.parameter.energy_guidance.learning_rate,
            )

            action_state_critic_optimizer = torch.optim.Adam(
                self.model["Policy"].critic.q.parameters(),
                lr=config.parameter.action_state_critic.learning_rate,
            )

            policy_optimizer = torch.optim.Adam(
                self.model["Policy"].policy.model.parameters(),
                lr=config.parameter.policy.learning_rate,
            )

            better_than_baseline = True
            baseline_return = -200

            for online_rl_iteration in track(range(config.parameter.online_rl.iterations), description="Online RL iteration"):

                def evaluate(model):
                    pass
                
                def collect(model, num_steps, random_policy=False, random_ratio=0.0):
                    if random_policy:
                        return self.simulator.collect_steps(policy=None, num_steps=num_steps, random_policy=True)
                    else:
                        def policy(obs: np.ndarray) -> np.ndarray:
                            obs = torch.tensor(obs, dtype=torch.float32, device=config.model.QGPOPolicy.device).unsqueeze(0)
                            action = model(obs).squeeze(0).cpu().detach().numpy()
                            # randomly replace some item of action with random action
                            if np.random.rand() < random_ratio:
                                # select random i from 0 to action.shape[0]
                                i = np.random.randint(0, action.shape[0])
                                # randomly select a value from -1 to 1
                                action[i] = np.random.rand() * 2 - 1
                            return action
                        return self.simulator.collect_steps(policy=policy, num_steps=num_steps)

                if better_than_baseline:
                    if online_rl_iteration > 0:
                        self.dataset.drop_data(config.parameter.online_rl.drop_ratio)
                        self.dataset.load_data(collect(self.model["Policy"], num_steps=config.parameter.online_rl.collect_steps, random_policy=False, random_ratio=0.01))
                    else:
                        self.dataset.load_data(collect(self.model["Policy"], num_steps=config.parameter.online_rl.collect_steps_at_the_beginning, random_policy=True))
                else:
                    self.dataset.load_data(collect(self.model["Policy"], num_steps=config.parameter.online_rl.collect_steps, random_ratio=0.2))

            #---------------------------------------
            # Customized training code ↑
            #---------------------------------------

            wandb.finish()


    def deploy(self, config:EasyDict = None):
        
        pass
