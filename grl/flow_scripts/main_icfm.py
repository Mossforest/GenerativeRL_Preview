# import gymnasium as gym
# import gym as gym
from typing import List, Optional, Tuple, Union

import os

import signal
import sys
import cv2
import h5py

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset

import treetensor

from PIL import Image

from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from matplotlib import animation

from grl.utils import set_seed
from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.utils.log import log
from grl.neural_network import register_module

from functools import partial


class H5PYDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size
        self.file_list = os.listdir(data_path)
        self.condition_length = 1

        # open all files and record the episode length by key "episode_length"
        self.episode_length = []
        # legal_episode_length is the episode length that is larger than condition_length
        self.legal_episode_length = []
        for file_name in self.file_list:
            if not file_name.endswith(".h5"):
                continue
            with h5py.File(os.path.join(data_path, file_name), "r") as f:
                self.episode_length.append(f["episode_length"][()])
                if f["episode_length"][()] > self.condition_length:
                    self.legal_episode_length.append(f["episode_length"][()]-self.condition_length+1)
                else:
                    print(f"episode length is {f['episode_length'][()]}, which is less than condition length {self.condition_length}")
                    self.legal_episode_length.append(0)
        
        # calculate the sum of legal episode length
        self.legal_episode_length_sum = []
        self.legal_episode_length_sum.append(0)
        for i in range(1, len(self.legal_episode_length)):
            self.legal_episode_length_sum.append(sum(self.legal_episode_length[:i]))

        self.total_length = sum(self.legal_episode_length)
        self.index = torch.zeros((self.total_length, 2), dtype=torch.int64)
        for i in range(len(self.legal_episode_length)):
            if self.legal_episode_length[i] > 0:
                self.index[self.legal_episode_length_sum[i]:self.legal_episode_length_sum[i] + self.legal_episode_length[i], 0] = i
                self.index[self.legal_episode_length_sum[i]:self.legal_episode_length_sum[i] + self.legal_episode_length[i], 1] = torch.arange(self.legal_episode_length[i])



    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_order, episode_order = self.index[idx]
        with h5py.File(os.path.join(self.data_path, self.file_list[file_order]), "r") as f:
            obs = torch.tensor(f["obs"][episode_order:episode_order + self.condition_length])
            numpy_obs = obs.cpu().numpy().astype("uint8")
            # slice every frame and resize to (image_size, image_size, 3)
            numpy_obs_list = [resize_image(numpy_obs[i], self.image_size) for i in range(self.condition_length)]
            obs = torch.tensor(numpy_obs_list)
            reward = torch.tensor(f["reward"][episode_order:episode_order + self.condition_length])
            action = torch.tensor(f["action"][episode_order:episode_order + self.condition_length])
            next_obs = torch.tensor(f["next_obs"][episode_order:episode_order + self.condition_length])
            numpy_next_obs = next_obs.cpu().numpy().astype("uint8")
            # slice every frame and resize to (image_size, image_size, 3)
            numpy_next_obs_list = [resize_image(numpy_next_obs[i], self.image_size) for i in range(self.condition_length)]
            next_obs = torch.tensor(numpy_next_obs_list)
            done = torch.tensor(f["done"][episode_order:episode_order + self.condition_length])

        return obs, reward, action, next_obs, done


def resize_image(image, size):
    # image is numpy array of shape (210, 160, 3), resize to (image_size, image_size, 3)
    return cv2.resize(image[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)

def transform_obs(obs):
    obs = obs - 128  # [0, 255] -> [-128, 127]
    obs = obs / 128  # [-128, 127] -> [-1, 1]
    # permute to (C, H, W)
    obs = torch.einsum("...hwc->...chw", obs)
    return obs

def reverse_transform_obs(obs):
    # permute to (H, W, C)
    obs = torch.einsum("...chw->...hwc", obs)
    obs = obs * 128 + 128  # [-1, 1] -> [0, 255]
    obs = torch.int8(obs)
    return obs


GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x

class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, to_cat: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs

class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        return x, d_outputs, u_outputs

class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNet(
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64], attn_depths=[0,0,0,0],
        )

        num_actions = 9
        cond_channels = 256
        num_steps_conditioning = 1

        self.conv_in = Conv3x3(3, 64)
        self.norm_out = GroupNorm(64)

        self.conv_out = Conv3x3(64, 3)

        self.act_emb = nn.Sequential(
            nn.Embedding(num_actions, cond_channels // num_steps_conditioning // 2),
            nn.Flatten(),  # b t e -> b (t e)
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        if condition is not None:
            action_emb = self.act_emb(condition.squeeze(1).to(torch.long))
            cond = torch.cat((action_emb, t), dim=1)
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        else:
            cond = t
        x = self.conv_in(x)
        x, d_outputs, u_outputs = self.unet(cond=cond, x=x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        x = x.unsqueeze(1)
        return x

register_module(MyModule, "MyModule")

def make_config(device):

    image_size = 256
    x_size = (1, 3, 256, 256)
    data_num=100000
    t_embedding_dim = 128
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
                image_size=image_size,
                image_path="/root/world_model_atari/atari_world_model/images/",
                data_path="/root/world_model_atari/atari_world_model/",
                origin_image_size=(210, 160, 3),
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
                        t_encoder=t_encoder,
                        backbone=dict(
                            type="MyModule",
                            args={},
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=5e-4,
                data_num=data_num,
                iterations=200000,
                batch_size=40,
                eval_freq=20,
                checkpoint_freq=20,
                checkpoint_path="/root/world_model_atari/atari_world_model/checkpoint-atari-world-model-icfm",
                video_save_path="/root/world_model_atari/atari_world_model/video-atari-world-model-icfm",
                device=device,
            ),
        )
    )

    return config

def pipeline(config):

    flow_model = IndependentConditionalFlowModel(config=config.flow_model).to(
        config.flow_model.device
    )
    flow_model = torch.compile(flow_model)

    dataset = H5PYDataset(config.dataset.data_path, config.dataset.image_size)

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

            # from collections import OrderedDict

            # checkpoint_sorted = OrderedDict()
            # for key, value in checkpoint["model"].items():
            #     name = key.replace("module.", "")
            #     checkpoint_sorted[name] = value

            flow_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    counter = 0
    iteration = 0

    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(12, 12))

        ims = []

        for i, data in enumerate(data_list):

            grid = make_grid(
                data.view([-1, 3, 256, 256]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=2
            )
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"{prefix}_{iteration}.mp4"),
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

    mp_list=[]
    optimizer.zero_grad()

    for iteration in range(config.parameter.iterations):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:

            # random sample a batch of data
            random_idx = torch.randint(0, len(dataset), (4,))
            obs_list = []
            action_list = []
            next_obs_list = []
            for idx in random_idx:
                obs, reward, action, next_obs, done = dataset[idx]
                obs = obs.to(config.device)
                obs = transform_obs(obs)
                obs_list.append(obs)
                action = action.to(config.device)
                action_list.append(action)
                next_obs = next_obs.to(config.device)
                next_obs = transform_obs(next_obs)
                next_obs_list.append(next_obs)
            obs = torch.cat(obs_list, dim=0).unsqueeze(1)
            action = torch.cat(action_list, dim=0).unsqueeze(-1)
            next_obs = torch.cat(next_obs_list, dim=0).unsqueeze(1)
            condition = treetensor.torch.tensor(dict(action=action, state=obs)).to(config.device)

            flow_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)

            x_t = (
                flow_model.sample_forward_process(t_span=t_span, x_0=obs, condition=condition)
                .cpu()
                .detach()
            )

            x_t = [
                x.squeeze(0).squeeze(1) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100, prefix="generate")

            #p = mp.Process(target=render_video, args=(x_t, config.parameter.video_save_path, iteration, 100, 100, "generate"))
            #p.start()
            #mp_list.append(p)

        sampler = torch.utils.data.RandomSampler(
                dataset, replacement=False
            )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.parameter.batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory=False,
            drop_last=True,
        )

        flow_model.train()

        for obs, reward, action, next_obs, done in track(data_loader, description=f"Epoch {iteration}"):
            obs = obs.to(config.device)
            obs = transform_obs(obs)
            action = action.to(config.device)
            next_obs = next_obs.to(config.device)
            next_obs = transform_obs(next_obs)

            if config.parameter.training_loss_type == "flow_matching":
                loss = flow_model.flow_matching_loss(x0=obs, x1=next_obs, condition=action)
            else:
                raise NotImplementedError("Unknown loss type")
            loss.backward()
            counter += 1
            optimizer.step()
            optimizer.zero_grad()
            

            log.info(
                f"iteration {iteration}, step {counter}, loss {loss.item()}"
            )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(flow_model, optimizer, iteration)


    for p in mp_list:
        p.join()



def main():
    
    device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
    config = make_config(device=device)

    log.info("config: \n{}".format(config))
    pipeline(config)

if __name__ == "__main__":
    main()
