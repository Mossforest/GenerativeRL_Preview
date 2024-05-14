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
        Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Arguments:
        patch_enabled (bool): if enabled, the input state should be [B, P, C] where the patch as a length of seq-model; 
                              if False, the input should be [B, C] and the cross attention will be [B, 1, C]
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
        self.cross_attention =  CrossAttention(input_size=hidden_size, num_heads=8, length_1=(not self.patch_enabled))
        self.classifer = MultiLayerPerceptron(
            hidden_sizes=[hidden_size] + [hidden_size for _ in range(classify_layer)],
            output_size=patch_size,
            activation=activation,
            final_activation=activation
        )
        # TODO: will multi-MLP tackle with the P dimension in [B, P, C]? I want it not to. where the only tackled dimention is C


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
        state_attn = state_attn.unsqueeze()
        predict = self.classifier(state_attn)
        return predict
    
    def loss(
        self,
        predict: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ):
        pass
        return loss