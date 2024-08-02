import os
import signal
import sys

from typing import Optional, Union

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.conditional_flow_model.optimal_transport_conditional_flow_model import (
    OptimalTransportConditionalFlowModel,
)
from grl.generative_models.metric import compute_likelihood
from grl.utils import set_seed
from grl.utils.log import log
from grl.neural_network import TemporalSpatialResidualNet, register_module
from torchcfm.models.unet import UNetModel


from grl.neural_network import MultiLayerPerceptron, get_module, register_module
from grl.neural_network.encoders import get_encoder, ExponentialFourierProjectionTimeEncoder, GaussianFourierProjectionEncoder
from grl.neural_network.transformers.dit import DiTBlock, FinalLayer1D, get_1d_pos_embed, modulate, get_2d_sincos_pos_embed, PatchEmbed, FinalLayer


class DiT2D_Special(nn.Module):
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
            learn_sigma (:obj:`bool`, defaults to True): Whether to learn sigma.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)
        # self.y_embedder = nn.Conv1d(in_channels=self.in_channels, out_channels=hidden_size, kernel_size=1, groups=in_channels, bias=True)
        self.y_embedder = GaussianFourierProjectionEncoder(embed_dim=hidden_size, x_shape=())
        self.y_embedder_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
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
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder_linear.weight, std=0.02)
        nn.init.normal_(self.y_embedder_linear.bias, std=0.02)

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
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """
        Overview:
            Forward pass of DiT.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of spatial inputs (images or latent representations of images).
            condition (:obj:`Union[torch.Tensor, TensorDict]`, optional): The input condition, such as class labels.
        """

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition.to(dtype=torch.float32))
            y= self.y_embedder_linear(y)
            # y.shape : (N, D)
            c = t + y  # (N, D)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


register_module(DiT2D_Special, "DiT2D_Special")

x_size = (1,28,28)
condition_size = 1
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

config = EasyDict(
    dict(
        device=device,
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
                sigma=0.0,
            ),
            model=dict(
                type="velocity_function",
                args=dict(
                    backbone = dict(
                        type = "DiT2D_Special",
                        args = dict(
                            input_size=28,
                            patch_size=2,
                            in_channels=1,
                            hidden_size=9*32,
                            depth=4,
                            num_heads=3,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=1e-3,
            iterations=20000,
            batch_size=128,
            clip_grad_norm=1.0,
            eval_freq=500,
            checkpoint_freq=1000,
            checkpoint_path="./checkpoint-mnist-dit-otcfm",
            video_save_path="./video-mnist-dit-otcfm",
            device=device,
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

    trainset = datasets.MNIST(
        "./test_scripts/mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.parameter.batch_size, shuffle=True, drop_last=True
    )


    #
    optimizer = torch.optim.Adam(
        flow_model.parameters(),
        lr=config.parameter.lr,
    )

    if config.parameter.checkpoint_path is not None:

        if (
            not os.path.exists(config.parameter.checkpoint_path)
            or len(os.listdir(config.parameter.checkpoint_path)) == 0
        ):
            log.warning(
                f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
            )
            last_iteration = -1
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.checkpoint_path)
                if f.endswith(".pt")
            ]
            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            checkpoint = torch.load(
                os.path.join(config.parameter.checkpoint_path, checkpoint_files[-1]),
                map_location="cpu",
            )
            flow_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1


    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_generator = get_train_data(data_loader)

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

        ims = []
        colors = np.linspace(0, 1, len(data_list))

        for i, data in enumerate(data_list):

            grid = make_grid(
                data.view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
            )
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"iteration_{iteration}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    def save_checkpoint(model, optimizer, iteration):
        if not os.path.exists(config.parameter.checkpoint_path):
            os.makedirs(config.parameter.checkpoint_path)
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                iteration=iteration,
            ),
            f=os.path.join(
                config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
            ),
        )

    history_iteration = [-1]

    def save_checkpoint_on_exit(model, optimizer, iterations):
        def exit_handler(signal, frame):
            log.info("Saving checkpoint when exit...")
            save_checkpoint(model, optimizer, iteration=iterations[-1])
            log.info("Done.")
            sys.exit(0)

        signal.signal(signal.SIGINT, exit_handler)

    save_checkpoint_on_exit(flow_model, optimizer, history_iteration)

    for iteration in track(range(config.parameter.iterations), description="Training"):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                flow_model.sample_forward_process(t_span=t_span, condition = torch.arange(10, device=config.device).repeat(10))
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100)

        batch_data = next(data_generator)
        x1 = batch_data[0].to(config.device)
        y = batch_data[1].to(config.device)

        # plot2d(batch_data.cpu().numpy())
        flow_model.train()
        if config.parameter.training_loss_type == "flow_matching":
            x0 = flow_model.gaussian_generator(x1.shape[0]).to(config.device)
            loss = flow_model.flow_matching_loss(x0=x0, x1=x1, condition=y)
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

        log.info(
            f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
        )

        history_iteration.append(iteration)

        if iteration == config.parameter.iterations - 1:
            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                flow_model.sample_forward_process(t_span=t_span, batch_size=500)
                .cpu()
                .detach()
            )
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(
                x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100
            )

        # if (iteration + 1) % config.parameter.checkpoint_freq == 0:
        #     save_checkpoint(flow_model, optimizer, iteration)
