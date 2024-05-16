import os
from easydict import EasyDict
from rich.progress import track
import numpy as np
import h5py
import time
import random
from typing import Union, Optional


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import wandb


from matplotlib import animation
from easydict import EasyDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import treetensor
from tensordict import TensorDict
from torch.utils.data import Dataset
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.neural_network import MultiLayerPerceptron, register_module
from grl.generative_models.metric import compute_likelihood

from timm.models.vision_transformer import PatchEmbed
from grl.neural_network.encoders import ExponentialFourierProjectionTimeEncoder
from grl.neural_network.transformers.dit import DiTBlock, FinalLayer, get_2d_sincos_pos_embed
from grl.cxy_models.diffusion import ConditionEmbedder, CrossAttention


class Classifier(nn.Module):
    """
    Overview:
        The delta-state classifier rho of model v2
    Arguments:
        patch_enabled (bool): if enabled, the input state should be [B, P, C] where the patch as a length of seq-model; 
                              if False, the input should be [B, C] and the cross attention will be [B, 1, C]
    Output:
        output (Tensor): if patch_enabled, the shape would be [B, P];
                         if else, the shape would be [B]
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            background_size: int,
            hidden_size: int,
            patch_enabled: bool = True,
            patch_size: int = None,
            state_layer: int = 4,
            action_layer: int = 4,
            classify_layer: int = 2,
            activation: str ="tanh",
        ):
        super().__init__()
        self.patch_enabled = patch_enabled
        self.patch_size = patch_size
        self.condition_embedder = ConditionEmbedder(action_size, background_size, hidden_size, layer=action_layer)
        self.state_embedder = MultiLayerPerceptron(
            hidden_sizes=[state_size] + [hidden_size for _ in range(state_layer)],
            output_size=hidden_size,
            activation=activation,
            final_activation=activation
        )
        self.cross_attention =  CrossAttention(input_size=hidden_size, num_heads=8)
        self.predicter = MultiLayerPerceptron(
            hidden_sizes=[hidden_size] + [hidden_size for _ in range(classify_layer)],
            output_size=1,
            activation=activation,
            final_activation=activation
        )
        # TODO: will multi-MLP tackle with the P dimension in [B, P, C]? I want it not to. where the only tackled dimension is C


    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            background: torch.Tensor,
        ):
        """
        Overview:
            Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
        Arguments:
            - state (Tensor): [B, P, C] if patch_enabled; else [B, C]
            - action (Tensor): [B, C]; will be extend to [B, P, C] by duplicate
            
        """
        condition = self.condition_embedder(action, background)
        state_emb = self.state_embedder(state)
        # cross-attention, where condition as query, state as key & value
        if not self.patch_enabled:
            state_emb = state_emb.unsqueeze(1)
            condition = condition.unsqueeze(1)
        else:
            assert state.shape[1] == self.patch_size
            condition = condition.unsqueeze(1).repeat(1, self.patch_size, 1)
        state_attn = self.cross_attention(condition, state_emb)
        state_attn = state_attn.squeeze()
        predict = self.predicter(state_attn).squeeze()
        return predict
    
    def loss(
        self,
        predict: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ):
        # euclid -> abs -> tanh to [0, 1]
        target = torch.abs(torch.norm(next_state - state, dim=-1))
        target = torch.tanh(target)
        loss = F.mse_loss(predict, target)
        return loss


if __name__ == '__main__':
    B = 64
    P = 4
    C = 16
    H = 256
    action_size = 4
    bg_size = 2
    enable_patch = True
    epoch = 100
    
    if enable_patch:
        state = torch.rand((B, P, C))
        next_state = torch.rand((B, P, C))
    else:
        state = torch.rand((B, C))
        next_state = torch.rand((B, C))
    action = torch.rand((B, action_size))
    background = torch.rand((B, bg_size))
    
    classifier = Classifier(C, action_size, bg_size, H, enable_patch, P)
    optim = torch.optim.SGD(classifier.parameters(), lr=0.01)
    
    for i in range(epoch):
        predict = classifier(state, action, background)
        loss = classifier.loss(predict, state, next_state)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'epoch {i}, loss {loss.item()}')
    
    print('done!')