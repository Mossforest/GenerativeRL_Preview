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
import torch.nn.functional as F
import treetensor
from tensordict import TensorDict
from torch.utils.data import Dataset
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.neural_network import MultiLayerPerceptron, register_module
from grl.generative_models.metric import compute_likelihood

from timm.models.vision_transformer import PatchEmbed
from grl.neural_network import get_module
from grl.neural_network.encoders import ExponentialFourierProjectionTimeEncoder
from grl.neural_network.transformers.dit import DiTBlock, FinalLayer, get_2d_sincos_pos_embed, get_1d_pos_embed, FinalLayer1D



class CrossAttention(nn.Module):
    """
    Overview:
        Implements cross-attention mechanism
    """
    def __init__(
        self, 
        input_size: int, 
        num_heads: int = 4,
        dropout: int = 0.1,
        ):
        super(CrossAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self, 
        Q: torch.Tensor,
        KV: torch.Tensor,
        key_padding_mask: bool=None
        ):
        """
        Arguments:
            - Q (Tensor[N, P, D]): Query embeddings with shape (batch_size, vec1_seq_length, input_size).
            - KV (Tensor[N, P, D]): Key & value embeddings with shape (batch_size, vec2_seq_length, input_size).
            - key_padding_mask (Tensor[N, K]): If provided, specifies which keys to skip attention to.
        Returns:
            - Tensor[N, Q, D]: The result of the cross-attention operation.
        """
        attention_output, _ = self.multihead_attention(query=Q, key=KV, value=KV, attn_mask=None, key_padding_mask=key_padding_mask)
        return attention_output


class ConditionEmbedder(nn.Module):
    """
    Overview:
        Embeds background information into action condition. Should be embed in model before the condition introducing as a embedder head
        module: action embedder & bg embedder & FilM
        input: action, background
        output: condition (B, cod_size)
    """
    def __init__(
            self,
            action_size: int,
            background_size: int,
            hidden_size: int,
            condition_size: int = None,
            layer: int = 4,
            activation: str ="tanh",
        ):
        super().__init__()
        if condition_size == None:
            condition_size = hidden_size
        self.action_embedder = MultiLayerPerceptron(
            hidden_sizes=[action_size] + [hidden_size for _ in range(layer)],
            output_size=condition_size,
            activation=activation,
            final_activation=activation,
        )
        self.background_embedder = nn.Linear(background_size, 2 * condition_size)
        # TODO: zero_init the background embedder

    def forward(
        self,
        action: torch.Tensor,
        background: torch.Tensor
        ):
        condition = self.action_embedder(action)
        out = self.background_embedder(background)
        # The dimension for splitting is 1 (condition dimension)
        gamma, beta = torch.split(out, out.shape[1] // 2, dim=1)
        condition = gamma * condition + beta
        return condition



class MLPEmbedder(nn.Module):
    """
    Overview:
        Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            layer: int = 4,
            activation: str ="tanh",
        ):
        super().__init__()
        self.input_size = input_size
        self.model = MultiLayerPerceptron(
            hidden_sizes=[input_size] + [hidden_size for _ in range(layer)],
            output_size=hidden_size,
            activation=activation,
            final_activation=activation,
        )

    def forward(
            self,
            x: torch.Tensor,
        ):
        x = self.model(x)
        return x


class DynamicDiT_1D(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for 1D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        token_size: int,  # == patch_size, here different input_channel in state_1d.shape can be view as the different patches
        in_channels: int = 4,  # == 1, the channel of each item in state (e.g. the img)
        action_size: int = 32,
        background_size: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
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
        self.out_channels = in_channels

        self.num_heads = num_heads

        self.x_embedder = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            groups=in_channels,
            bias=False,
        )
        self.y_embedder = ConditionEmbedder(action_size, background_size, hidden_size)
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_1d_pos_embed(embed_dim=hidden_size, grid_num=token_size)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
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
        # breakpoint()

        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        # x is of shape (N, T, C), reshape to (N, C, T)
        x = torch.einsum("ntc->nct", x)
        x = self.x_embedder(x) + torch.einsum("th->ht", self.pos_embed)

        t = self.t_embedder(t)  # (N, hidden_size)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition['action'], condition['dynamic'])  # (N, hidden_size)
            c = t + y  # (N, hidden_size)
        else:
            c = t
        
        x = torch.einsum("nht->nth", x)
        for block in self.blocks:
            x = block(x, c)  # (N, total_patches, hidden_size)
        x = self.final_layer(x, c)  # (N, total_patches, C)
        return x


class DynamicDiT(nn.Module):
    """
    Overview:
        Diffusion model with a Transformer backbone.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
        self,
        input_size: int = 32,
        action_size: int = 32,
        background_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            input_size (:obj:`int`, defaults to 32): The input size.
            patch_size (:obj:`int`, defaults to 2): The patch size.
            in_channels (:obj:`int`, defaults to 4): The number of input channels.
            hidden_size (:obj:`int`, defaults to 1152): The hidden size.
            depth (:obj:`int`, defaults to 28): The depth.
            num_heads (:obj:`int`, defaults to 16): The number of attention heads.
            mlp_ratio (:obj:`float`, defaults to 4.0): The hidden size of the MLP with respect to the hidden size of Attention.
            class_dropout_prob (:obj:`float`, defaults to 0.1): The class dropout probability.
            num_classes (:obj:`int`, defaults to 1000): The number of classes.
            learn_sigma (:obj:`bool`, defaults to True): Whether to learn sigma.
        """
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.action_size = action_size
        self.background_size = background_size
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        self.y_embedder = ConditionEmbedder(action_size, background_size, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

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

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            imgs (:obj:`torch.Tensor`): The output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): (N, T, patch_size**2 * C)
            imgs (:obj:`torch.Tensor`): (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        ):
        """
        Overview:
            Forward pass of DiT.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        if condition is not None:
            y = self.y_embedder(condition['action'], condition['dynamic'])    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


class DiT_Special(nn.Module):
    """
    Overview:
        Diffusion model with a Transformer backbone.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        y_input_size: int = 32,
        learn_sigma: bool = True,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            input_size (:obj:`int`, defaults to 32): The input size.
            patch_size (:obj:`int`, defaults to 2): The patch size.
            in_channels (:obj:`int`, defaults to 4): The number of input channels.
            hidden_size (:obj:`int`, defaults to 1152): The hidden size.
            depth (:obj:`int`, defaults to 28): The depth.
            num_heads (:obj:`int`, defaults to 16): The number of attention heads.
            mlp_ratio (:obj:`float`, defaults to 4.0): The hidden size of the MLP with respect to the hidden size of Attention.
            class_dropout_prob (:obj:`float`, defaults to 0.1): The class dropout probability.
            num_classes (:obj:`int`, defaults to 1000): The number of classes.
            learn_sigma (:obj:`bool`, defaults to True): Whether to learn sigma.
        """
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        self.y_embedder = MLPEmbedder(y_input_size, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # TODO: Initialize MLP embedding:
        for net in self.y_embedder.model.model:
            try:
                nn.init.normal_(net.weight, std=0.02)
            except AttributeError:
                pass

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

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the input tensor.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            imgs (:obj:`torch.Tensor`): The output tensor.
        Shapes:
            x (:obj:`torch.Tensor`): (N, T, patch_size**2 * C)
            imgs (:obj:`torch.Tensor`): (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        ):
        """
        Overview:
            Forward pass of DiT.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        
        if condition is not None:
            y = self.y_embedder(condition)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            condition: Optional[Union[torch.Tensor, TensorDict]] = None,
            cfg_scale: float = 1.0,):
        """
        Overview:
            Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
            cfg_scale (:obj:`float`, defaults to 1.0): The scale for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, condition)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


if __name__ == '__main__':
    B = 64
    P = 4
    C = 16
    H = 256
    action_size = 4
    bg_size = 2
    enable_patch = False
    epoch = 100
    
    if enable_patch:
        state = torch.rand((B, P, C))
        next_state = torch.rand((B, P, C))
        target = torch.rand((B, P, C))
    else:
        state = torch.rand((B, C))
        next_state = torch.rand((B, C))
        target = torch.rand((B, C))
    action = torch.rand((B, action_size))
    background = torch.rand((B, bg_size))
    condition = {'action': action, 'dynamic': background}
    t = torch.rand((B,))
    
    if enable_patch:
        model = DynamicDiT_1D(P, C, action_size, bg_size, H)
    else:
        model = DynamicDiT_1D(C, 1, action_size, bg_size, H)
    optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    
    for i in range(epoch):
        predict = model(t, state, condition).squeeze()
        loss = F.mse_loss(predict, target)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'epoch {i}, loss {loss.item()}')
    
    print('done!')