import math
from typing import Callable, List, Optional, Union
from tensordict import TensorDict

import numpy as np
import torch
import torch.nn as nn
from grl.neural_network.activation import get_activation


# from grl.neural_network import register_module
class  TensorDictencoder(torch.nn.Module):
    def __init__(self):
        super(TensorDictencoder, self).__init__()

    def forward(self, x: dict) -> torch.Tensor:
        tensors = []
        for v in x.values():
            if v.dim() == 3 and v.shape[0] == 1:
                v = v.view(1, -1)
            tensors.append(v)
        x = torch.cat(tensors, dim=1)
        return x


def get_encoder(type: str):
    """
    Overview:
        Get the encoder module by the encoder type.
    Arguments:
        type (:obj:`str`): The encoder type.
    """

    if type.lower() in ENCODERS:
        return ENCODERS[type.lower()]
    else:
        raise ValueError(f"Unknown encoder type: {type}")


class GaussianFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding time variable.
        This module is used as the encoder of time in generative models such as diffusion model.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

            \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B,)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, embed_dim)`, where B is batch size, embed_dim is the \
                dimension of the output embedding vector.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(100)
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierProjectionEncoder(nn.Module):
    r"""
    Overview:
        Gaussian random features for encoding variables.
        This module can be seen as a generalization of GaussianFourierProjectionTimeEncoder for encoding multi-dimensional variables.
        It transforms the input tensor :math:`x` to a high-dimensional embedding vector :math:`\phi(x)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(x) = [ \sin(x \cdot w_1), \cos(x \cdot w_1), \sin(x \cdot w_2), \cos(x \cdot w_2), \ldots, \sin(x \cdot w_{\text{embed\_dim} / 2}), \cos(x \cdot w_{\text{embed\_dim} / 2}) ]

        where :math:`w_i` is a random scalar sampled from the Gaussian distribution.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, embed_dim, x_shape, flatten=True, scale=30.0):
        """
        Overview:
            Initialize the Gaussian Fourier Projection Time Encoder according to arguments.
        Arguments:
            embed_dim (:obj:`int`): The dimension of the output embedding vector.
            x_shape (:obj:`tuple`): The shape of the input tensor.
            flatten (:obj:`bool`): Whether to flatten the output tensor afyer applying the encoder.
            scale (:obj:`float`): The scale of the Gaussian random features.
        """
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale * 2 * np.pi, requires_grad=False
        )
        self.x_shape = x_shape
        self.flatten = flatten

    def forward(self, x):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            x (:obj:`torch.Tensor`): Input time step tensor.
        Returns:
            output (:obj:`torch.Tensor`): Output embedding vector.
        Shapes:
            x (:obj:`torch.Tensor`): :math:`(B, D)`, where B is batch size.
            output (:obj:`torch.Tensor`): :math:`(B, D * embed_dim)` if flatten is True, otherwise :math:`(B, D, embed_dim)`.
                where B is batch size, embed_dim is the dimension of the output embedding vector, D is the shape of the input tensor.
        Examples:
            >>> encoder = GaussianFourierProjectionTimeEncoder(128)
            >>> x = torch.randn(torch.Size([100, 10]))
            >>> output = encoder(x)
        """
        x_proj = x[..., None] * self.W[None, :]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # if x shape is (B1, ..., Bn, **x_shape), then the output shape is (B1, ..., Bn, np.prod(x_shape) * embed_dim)
        if self.flatten:
            x_proj = torch.flatten(x_proj, start_dim=-1 - self.x_shape.__len__())

        return x_proj


class ExponentialFourierProjectionTimeEncoder(nn.Module):
    r"""
    Overview:
        Expoential Fourier Projection Time Encoder.
        It transforms the time :math:`t` to a high-dimensional embedding vector :math:`\phi(t)`.
        The output embedding vector is computed as follows:

        .. math::

                \phi(t) = [ \sin(t \cdot w_1), \cos(t \cdot w_1), \sin(t \cdot w_2), \cos(t \cdot w_2), \ldots, \sin(t \cdot w_{\text{embed\_dim} / 2}), \cos(t \cdot w_{\text{embed\_dim} / 2}) ]

            where :math:`w_i` is a random scalar sampled from a uniform distribution, then transformed by exponential function.
        There is an additional MLP layer to transform the frequency embedding:

        .. math::

            \text{MLP}(\phi(t)) = \text{SiLU}(\text{Linear}(\text{SiLU}(\text{Linear}(\phi(t)))))

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Overview:
            Initialize the timestep embedder.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            frequency_embedding_size (:obj:`int`): The size of the frequency embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    # TODO: simplify this function
    @staticmethod
    def timestep_embedding(t, embed_dim, max_period=10000):
        """
        Overview:
            Create sinusoidal timestep embeddings.
        Arguments:
            t (:obj:`torch.Tensor`): a 1-D Tensor of N indices, one per batch element. These may be fractional.
            embed_dim (:obj:`int`): the dimension of the output.
            max_period (:obj:`int`): controls the minimum frequency of the embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embed_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        """
        Overview:
            Return the output embedding vector of the input time step.
        Arguments:
            t (:obj:`torch.Tensor`): Input time step tensor.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPEncoder(nn.Module):
    # ./grl/neural_network/__init__.py#L365

    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        activation: Union[str, List[str]],
        dropout: float = None,
        layernorm: bool = False,
        final_activation: str = None,
        scale: float = None,
        shrink: float = None,
    ):
        super().__init__()

        self.model = nn.Sequential()

        for i in range(len(hidden_sizes) - 1):
            self.model.add_module(
                "linear" + str(i), nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )

            if isinstance(activation, list):
                self.model.add_module(
                    "activation" + str(i), get_activation(activation[i])
                )
            else:
                self.model.add_module("activation" + str(i), get_activation(activation))
            if dropout is not None and dropout > 0:
                self.model.add_module("dropout", nn.Dropout(dropout))
            if layernorm:
                self.model.add_module("layernorm", nn.LayerNorm(hidden_sizes[i + 1]))

        self.model.add_module(
            "linear" + str(len(hidden_sizes) - 1),
            nn.Linear(hidden_sizes[-1], output_size),
        )

        if final_activation is not None:
            self.model.add_module("final_activation", get_activation(final_activation))

        if scale is not None:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)
        else:
            self.scale = 1.0

        # shrink the weight of linear layer 'linear'+str(len(hidden_sizes) to it's origin 0.01
        if shrink is not None:
            if final_activation is not None:
                self.model[-2].weight.data.normal_(0, shrink)
            else:
                self.model[-1].weight.data.normal_(0, shrink)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the output of the multi-layer perceptron.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        return self.scale * self.model(x)


class MultiMLPEncoder(nn.Module):
    # for dynamic model use: encode action and background seperately

    def __init__(self, **kwargs):
        super().__init__()

        self.model = torch.nn.ModuleDict()
        for key, value in kwargs.items():
            if "encoder" in key:
                self.model[key] = MLPEncoder(**value) # TODO: **config.t_encoder.args, same thing?

    def forward(self, x: TensorDict) -> torch.Tensor:
        """
        Overview:
            Return the output of the multi-layer perceptron.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        """
        outputs = TensorDict()
        for key, value in x.items():
            outputs[key] = self.model[f'{key}_encoder'](value)
        return outputs


ENCODERS = {
    "GaussianFourierProjectionTimeEncoder".lower(): GaussianFourierProjectionTimeEncoder,
    "GaussianFourierProjectionEncoder".lower(): GaussianFourierProjectionEncoder,
    "ExponentialFourierProjectionTimeEncoder".lower(): ExponentialFourierProjectionTimeEncoder,
    "SinusoidalPosEmb".lower(): SinusoidalPosEmb,
    "TensorDictencoder".lower(): TensorDictencoder,
    "MLPEncoder".lower(): MLPEncoder,
    "MultiMLPEncoder".lower(): MultiMLPEncoder,
}
