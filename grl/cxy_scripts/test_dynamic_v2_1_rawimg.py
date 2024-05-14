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


project="test_dynamic_v2_1_rawimg"


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
            #TODO: polish this part
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

register_module(DiT_Special, "DiT_Special")



class MetadriveDataset(Dataset):
    def __init__(self, data_list, key_list):
        self.data_list, self.key_list = self.normalize(data_list, key_list)

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        return {key: item[idx] for key, item in zip(self.key_list, self.data_list)}
    
    def normalize(self, data_list, key_list):
        # transverse data to [-1, 1] via activation func
        # action: [-1, 1] naturally
        # state: go tanh
        # dynamics: fixed [min, max] -> [-1, 1]
        #   max_engine_force = [0, 1000]
        #   max_brake_force = [0, 200]
        #   wheel_friction = [0, 1]
        #   max_steering = [0, 90]
        #   max_speed = [0, 100]
        
        # state
        state = data_list[key_list.index('state')]
        data_list[key_list.index('state')] = torch.tanh(state).float()
        next_state = data_list[key_list.index('next_state')]
        data_list[key_list.index('next_state')] = torch.tanh(next_state).float()
        
        # dynamic (* the first elem should be separated into 'dynamic_type')
        dynamic = data_list[key_list.index('dynamic')]
        dyna_max = torch.tensor([1000, 200, 1, 90, 100]).to(dynamic.device)
        dyna_min = torch.tensor([0, 0, 0, 0, 0]).to(dynamic.device)
        
        dynamic_type = dynamic[:, 0]
        key_list.append('dynamic_type')
        data_list.append(dynamic_type)
        
        dyna = dynamic[:, 1:]
        dyna = (dyna_max - dyna) / (dyna_max - dyna_min) * 2 - 1
        data_list[key_list.index('dynamic')] = dyna.float()
        
        # prepare condition = action + dynamic
        action = data_list[key_list.index('action')]
        condition = torch.cat((action, dyna), dim=1).float()
        key_list.append('condition')
        data_list.append(condition)
        
        # prepare blank_condition for CFG & v21
        blank_condition = torch.zeros_like(condition).to(condition.device).float()
        key_list.append('blank_condition')
        data_list.append(blank_condition)
        
        return data_list, key_list


def get_dataset(data_path1, data_path2, train_ratio=0.9):

    data_dict = torch.load(data_path1)
    data_list = list(data_dict.items())
    key_list = [item[0] for item in data_list]
    value_list = [item[1] for item in data_list]
    
    data_dict2 = torch.load(data_path2)
    data_list2 = list(data_dict2.items())
    key_list2 = [item[0] for item in data_list2]
    value_list2 = [item[1] for item in data_list2]

    np.random.seed(0)  # 为了可重现性设置随机种子
    permuted_indices = np.random.permutation(len(value_list[0]))
    train_list = [item[permuted_indices] for item in value_list]
    
    permuted_indices2 = np.random.permutation(len(value_list2[0]))
    eval_list = [item[permuted_indices2] for item in value_list2]
    
    train_dataset = MetadriveDataset(train_list, key_list)
    eval_dataset = MetadriveDataset(eval_list, key_list)
    
    return train_dataset, eval_dataset


def save_checkpoint(model, optimizer, iteration, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(
            checkpoint_path,
            f"checkpoint_{iteration}.pt"
        )
    )





sweep_config = EasyDict(
    name=project,
    metric=dict(
        name="loss",
        goal="minimize",
    ),
    method="grid",
    parameters=dict(
        diffusion_model=dict(
            parameters=dict(
                path=dict(
                    parameters=dict(
                        type=dict(values=["gvp"], ),
                    ),
                ),
                model=dict(
                    parameters=dict(
                        type=dict(
                            values=[
                                "velocity_function",
                            ],
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            parameters=dict(
                training_loss_type=dict(
                    values=["flow_matching"],
                ),
                lr=dict(
                    values=[1e-4] # ,3e-3,4e-3,5e-3],
                ),
            ),
        ),
    ),
)


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config = EasyDict(
        dict(
            x_size = (5, 84, 84), # (64, 6, 6),
            device = device,
            diffusion_model = dict(
                device = device,
                x_size = (5, 84, 84),
                alpha = 1.0,
                solver = dict(
                    type = "ODESolver",
                    args = dict(
                        library="torchdyn",
                    ),
                ),
                path = dict(
                    type = "gvp", 
                    beta_0 = 0.1,
                    beta_1 = 20.0,
                ),
                model = dict(
                    type = "score_function", # "velocity_function",
                    args = dict(
                        backbone = dict(
                            type = "DiT_Special",
                            args = dict(
                                input_size = 84,
                                patch_size = 2,
                                in_channels = 5,
                                hidden_size = 256,
                                depth = 6,
                                num_heads = 8,
                                mlp_ratio = 2.0,
                                y_input_size=7,  # [action, dynamic]
                                learn_sigma = False,
                            ),
                        ),
                    ),
                ),
            ),
            parameter = dict(
                training_loss_type = "score_matching", # "flow_matching",
                lr=5e-4,
                weight_decay=0,
                iterations=100000,
                batch_size=64,
                clip_grad_norm=1.0,
                eval_freq=5000,
                eval_batch_size=16, #500,
                device=device,
            ),
            data_path = '/mnt/nfs/chenxinyan/muzero/muzero_hidden_data/ver2_0430/data/data_dict_2063.pth',
            data_path2 = '/mnt/nfs/chenxinyan/muzero/muzero_hidden_data/ver2_0430/data/data_dict2_1090.pth',
    ))

    with wandb.init(
            project=project,
            config=config,
    ) as wandb_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = EasyDict(wandb.config)
        run_name = 'v21_conddit_0430'
        wandb.run.name = run_name
        wandb.run.save()
        
        
        # get data
        def get_train_data(dataloader):
            while True:
                yield from dataloader
        dataset, eval_dataset = get_dataset(config.data_path, config.data_path2)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.parameter.batch_size, shuffle=True)
        data_generator = get_train_data(data_loader)
        eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.parameter.eval_batch_size, shuffle=False)
        eval_data_generator = get_train_data(eval_data_loader)

        diffusion_model = DiffusionModel(config=config.diffusion_model).to(config.diffusion_model.device)
        wandb_run.watch(diffusion_model, log='all')


        # optim
        optimizer = torch.optim.Adam(
            diffusion_model.parameters(), 
            lr=config.parameter.lr,
            weight_decay=config.parameter.weight_decay,
            )

        gradient_sum=0.0
        loss_sum=0.0
        counter=0
        iteration=0
        eval_batch_data = next(eval_data_generator)
        
        for iteration in track(range(config.parameter.iterations), description="Training"):

            batch_data = next(data_generator)
            
            diffusion_model.train()
            if config.parameter.training_loss_type=="flow_matching":
                loss=diffusion_model.flow_matching_loss(x=batch_data['next_state'], condition=batch_data['blank_condition'])
            elif config.parameter.training_loss_type=="score_matching":
                loss=diffusion_model.score_matching_loss(x=batch_data['next_state'], condition=batch_data['blank_condition'])
            else:
                raise NotImplementedError("Unknown loss type")
            if torch.isnan(loss):
                print(f'+++++++++++++++++ loss is nan at {iteration}')
                breakpoint()
            
            optimizer.zero_grad()
            loss.backward()
            gradien_norm = torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), config.parameter.clip_grad_norm)
            optimizer.step()
            gradient_sum+=gradien_norm.item()
            loss_sum+=loss.item()
            counter+=1

            print(f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}")
            
            if (iteration > 0 and iteration % config.parameter.eval_freq == 0) or iteration == config.parameter.iterations - 1:
                t1 = time.time()
                logp=compute_likelihood(
                    model=diffusion_model,
                    x=eval_batch_data['next_state'],
                    using_Hutchinson_trace_estimator=True)
                print(f'finish eval: {time.time() - t1}')
                logp_mean = logp.mean()
                bits_per_dim = -logp_mean / (torch.prod(torch.tensor(config.x_size, device=config.device)) * torch.log(torch.tensor(2.0, device=config.device)))
                print(f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}, log likelihood {logp_mean.item()}, bits_per_dim {bits_per_dim.item()}")
                
                save_checkpoint(diffusion_model, optimizer, iteration, f'{project}/{run_name}/checkpoint')
                wandb_run.log(data=dict(iteration=iteration, gradient=gradient_sum / counter, loss=loss.item(), log_likelihood=logp_mean.item(), bits_per_dim=bits_per_dim.item()), commit=True)
            else:
                wandb_run.log(data=dict(iteration=iteration, gradient=gradient_sum / counter, loss=loss.item()), commit=True)

if __name__ == '__main__':
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=project
    )
    wandb.agent(sweep_id, function=main)
