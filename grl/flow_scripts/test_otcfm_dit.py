import multiprocessing as mp
import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.metric import compute_likelihood
from grl.rl_modules.value_network.one_shot_value_function import OneShotValueFunction
from grl.utils import set_seed
from grl.utils.log import log

from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.neural_network import MultiLayerPerceptron, get_module, register_module
from grl.neural_network.encoders import get_encoder, ExponentialFourierProjectionTimeEncoder, GaussianFourierProjectionEncoder
from grl.neural_network.transformers.dit import DiTBlock, FinalLayer1D, get_1d_pos_embed, modulate, get_2d_sincos_pos_embed, PatchEmbed, FinalLayer


class DiT1D_Special(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for 1D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        x_size: int,
        condition_size: int,
        in_channels: int = 4,
        out_channels: int = None,
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

        self.x_size = x_size
        self.condition_size = condition_size

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.num_heads = num_heads

        self.x_embedder = nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=1, groups=in_channels, bias=False)
        self.x_embedder_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.y_embedder = nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=1, groups=in_channels, bias=False)
        self.y_embedder_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_1d_pos_embed(embed_dim=hidden_size, grid_num=x_size+condition_size)
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
        wy = self.y_embedder.weight.data
        nn.init.xavier_uniform_(wy.view([wy.shape[0], -1]))

        # Initialize x_embedder_linear and y_embedder_linear:
        nn.init.xavier_uniform_(self.x_embedder_linear.weight)
        nn.init.normal_(self.x_embedder_linear.bias, std=1e-6)
        nn.init.xavier_uniform_(self.y_embedder_linear.weight)
        nn.init.normal_(self.y_embedder_linear.bias, std=1e-6)

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
            condition: torch.Tensor = None,
        ):
        """
        Overview:
            Forward pass of DiT for 3D data.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of inputs with spatial information (originally at t=0 it is tensor of videos or latent representations of videos).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """


        x = torch.cat([condition, x], dim=-1).unsqueeze(1)
        x = self.x_embedder(x)
        x = torch.einsum("nht->nth", x)
        x = self.x_embedder_linear(x)

        x = x + self.pos_embed
        t = self.t_embedder(t)                   # (N, hidden_size)
        c = t
        for block in self.blocks:
            x = block(x, c)                      # (N, T+TC, hidden_size)
        x = self.final_layer(x, c)                # (N, T+TC, C)
        x = x[:, self.condition_size:].squeeze(-1)                # (N, T, C)
        return x

register_module(DiT1D_Special, "DiT1D_Special")


x_size = 2
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
config = EasyDict(
    dict(
        device=device,
        dataset=dict(
            data_num=10000,
            noise=0.6,
            temperature=0.1,
        ),
        flow_model=dict(
            device=device,
            x_size=x_size,
            alpha=1.0,
            solver=dict(
                type="ODESolver",
                args=dict(
                    library="torchdyn",
                ),
            ),
            path=dict(
                sigma=0.1,
            ),
            model=dict(
                type="velocity_function",
                args=dict(
                    backbone=dict(
                        type="DiT1D_Special",
                        args=dict(
                            x_size = x_size,
                            condition_size = 1,
                            in_channels = 1,
                            out_channels = 1,
                            hidden_size = 64*6,
                            depth = 3,
                            num_heads = 4,
                            mlp_ratio = 4.0,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=1e-3,
            iterations=200000,
            batch_size=2048,
            clip_grad_norm=1.0,
            device=device,
            evaluation=dict(
                eval_freq=200,
                video_save_path="./video-swiss-roll-otcfm-dit",
                model_save_path="./model-swiss-roll-otcfm-dit",
                guidance_scale=[0, 1, 2, 4, 8, 16],
            ),
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")

    flow_model = OptimalTransportConditionalFlowModel(config=config.flow_model).to(
        config.flow_model.device
    )

    flow_model = torch.compile(flow_model)

    # get data
    x_and_t = make_swiss_roll(
        n_samples=config.dataset.data_num, noise=config.dataset.noise
    )
    t = x_and_t[1].astype(np.float32)
    value = (t - np.min(t)) / (np.max(t) - np.min(t))
    x = x_and_t[0].astype(np.float32)[:, [0, 2]]
    # transform data
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 10 - 5

    # plot data with color of value
    plt.scatter(x[:, 0], x[:, 1], c=value, vmin=-5, vmax=3)
    plt.colorbar()
    if not os.path.exists(config.parameter.evaluation.video_save_path):
        os.makedirs(config.parameter.evaluation.video_save_path)
    plt.savefig(
        os.path.join(
            config.parameter.evaluation.video_save_path, f"swiss_roll_data.png"
        )
    )
    plt.clf()

    diffusion_model_iteration = 0
    data = np.concatenate([x, value[:, None]], axis=1)
    #
    optimizer = torch.optim.Adam(
        flow_model.parameters(),
        lr=config.parameter.lr,
    )

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.batch_size, shuffle=True
    )

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_generator = get_train_data(data_loader)
    moving_average_v_loss = 0.0
    gradient_sum = 0.0
    loss_sum = 0.0
    counter = 0
    iteration = 0

    def plot2d(data):

        plt.scatter(data[:, 0], data[:, 1])
        plt.show()

    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        ims = []
        colors = np.linspace(0, 1, len(data_list))

        for i, data in enumerate(data_list):
            # image alpha frm 0 to 1
            im = plt.scatter(data[:, 0], data[:, 1], s=1)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
        ani.save(
            os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()


    data_generator = get_train_data(data_loader)
    subprocess_list = []
    for iteration in track(range(config.parameter.iterations), description="Training"):


        if iteration % 500 == 0:
            flow_model.eval()
            tensor = torch.full((500, 1), 0.88).to(config.device)
            t_span = torch.linspace(0.0, 1.0, 1000).to(config.device)
            x_t = (
                (flow_model.sample_forward_process(t_span=t_span, condition=tensor))
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            p = mp.Process(
                target=render_video,
                args=(
                    x_t,
                    config.parameter.evaluation.video_save_path,
                    iteration,
                    100,
                    100,
                ),
            )
            p.start()
            subprocess_list.append(p)

        batch_data = next(data_generator)
        batch_data = batch_data.to(config.device)
        train_x, train_value = batch_data[:, :x_size], batch_data[:, x_size]
        condition = batch_data[:, x_size:]
        # plot2d(batch_data.cpu().numpy())
        flow_model.train()
        if config.parameter.training_loss_type == "flow_matching":
            x0 = flow_model.gaussian_generator(train_x.shape[0]).to(config.device)
            loss = flow_model.flow_matching_loss(
                x0=x0, x1=train_x, condition=condition
            )
        else:
            raise NotImplementedError("Unknown loss type")
        optimizer.zero_grad()
        loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            flow_model.parameters(), config.parameter.clip_grad_norm
        )
        optimizer.step()
        gradient_sum += gradien_norm.item()
        loss_sum += loss.item()
        counter += 1
        flow_model_iteration = iteration
        log.info(
            f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
        )

        a = 0
        if a > 0:
            for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                tensor = torch.full((500, 1), i).to(config.device)
                t_span = torch.linspace(0.0, 1.0, 1000).to(config.device)
                x_t = (
                    (flow_model.sample_forward_process(t_span=t_span, condition=tensor))
                    .cpu()
                    .detach()
                )
                x_t = [x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)]
                render_video(
                    x_t,
                    config.parameter.evaluation.video_save_path,
                    i,
                    100,
                    100,
                )

    for p in subprocess_list:
        p.join()
