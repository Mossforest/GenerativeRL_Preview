from typing import Callable, List, Optional, Union, Tuple
from easydict import EasyDict
import math
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from grl.neural_network import MLP, get_module
from grl.neural_network.encoders import get_encoder
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def get_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = np.einsum('...,d->...d', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb


def get_1d_pos_embed(
        embed_dim,
        grid_num
    ):
    """
    Overview:
        Get 1D positional embeddings for 3D data.
    Arguments:
        - embed_dim (:obj:`int`): The output dimension of embeddings for each grid.
        - grid_num (:obj:`int`): The number of the grid in each dimension.
    """

    # grid = np.arange(grid_num[0], dtype=np.float32)
    grid = np.linspace(0, grid_num, num=grid_num)

    emb = get_sincos_pos_embed_from_grid(embed_dim, grid)

    return emb

class TimestepEmbedder(nn.Module):
    """
    Overview:
        Embeds scalar timesteps into vector representations.
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Overview:
            Initialize the timestep embedder.
        Arguments:
            - hidden_size (:obj:`int`): The hidden size.
            - frequency_embedding_size (:obj:`int`): The size of the frequency embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    #TODO: simplify this function
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionNetwork(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict()
        if hasattr(config, "action_encoder"):
            self.model["action_encoder"] = get_encoder(config.action_encoder.type)(**config.action_encoder.args)
        else:
            self.model["action_encoder"] = torch.nn.Identity()
        if hasattr(config, "state_encoder"):
            self.model["state_encoder"] = get_encoder(config.state_encoder.type)(**config.state_encoder.args)
        else:
            self.model["state_encoder"] = torch.nn.Identity()
        #TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(**config.backbone.args)


    def forward(
        self,
        action: Union[torch.Tensor, TensorDict],
        state: Union[torch.Tensor, TensorDict],
    ) -> torch.Tensor:
        """
        Overview:
            Return output of Q networks.
        Arguments:
            - action (:obj:`Union[torch.Tensor, TensorDict]`): The input action.
            - state (:obj:`Union[torch.Tensor, TensorDict]`): The input state.
        Returns:
            - q (:obj:`Union[torch.Tensor, TensorDict]`): The output of Q network.
        """
        action_embedding = self.model["action_encoder"](action)
        state_embedding = self.model["state_encoder"](state)
        return self.model["backbone"](action_embedding, state_embedding)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Modulate the input tensor x with the shift and scale tensors.
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor.
        - shift (:obj:`torch.Tensor`): The shift tensor.
        - scale (:obj:`torch.Tensor`): The scale tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    Overview:
        A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        """
        Overview:
            Initialize the DiT block.
        Arguments:
            - hidden_size (:obj:`int`): The hidden size.
            - num_heads (:obj:`int`): The number of attention heads.
            - mlp_ratio (:obj:`float`, defaults to 4.0): The hidden size of the MLP with respect to the hidden size of Attention.
            - block_kwargs (:obj:`dict`): The keyword arguments for the attention block.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Overview:
        The final layer of DiT.
        This is the official implementation of Github repo:
        https://github.com/facebookresearch/DiT/blob/main/models.py
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(self, hidden_size, out_channels):
        """
        Overview:
            Initialize the final layer.
        Arguments:
            - hidden_size (:obj:`int`): The hidden size.
            - out_channels (:obj:`int`): The number of output channels.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class transformer_1d(nn.Module):
    
    def __init__(self, input_dim, sequence_dim, hidden_dim, output_dim, condition_config, condition_dim=None, num_heads=4, mlp_ratio=4.0, depth=6):
        super(transformer_1d, self).__init__()
        self.input_dim=input_dim
        self.sequence_dim=sequence_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        
        self.x_embedder=nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        
        self.pos_embed=nn.Parameter(torch.tensor(get_1d_pos_embed(self.hidden_dim,self.sequence_dim), dtype=torch.float32), requires_grad=False)
        self.t_embedder=TimestepEmbedder(self.hidden_dim)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


        self.y_embedder=ConditionNetwork(condition_config)

        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)
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
        
        x = x.unsqueeze(-1)
        # x: (B, T=24, D=1)
        x = self.x_embedder(x) + self.pos_embed
        # x: (B, T, H=128)
        
        # t: (B, 32)
        t = self.t_embedder(t)
        # t: (B, H)
        
        if condition is not None:
            #TODO: polish this part
            y = self.y_embedder(**condition)    # (N, D)
            c = t + y                         # (N, D)
        else:
            c = t
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        
        # x: (B, T, H=256)
        x = self.final_layer(x, c)
            
        return x.squeeze(-1)
