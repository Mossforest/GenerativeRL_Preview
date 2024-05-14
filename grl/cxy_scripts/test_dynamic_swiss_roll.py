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

from grl.cxy_models.diffusion import DiT_Special
from grl.cxy_models.datasets import MetadriveDataset, get_dataset

project="test_dynamic_swiss_roll"

register_module(DiT_Special, "DiT_Special")


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
            x_size = (64, 6, 6),
            device = device,
            diffusion_model = dict(
                device = device,
                x_size = (64, 6, 6),
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
                                input_size = 6,
                                patch_size = 2,
                                in_channels = 64,
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
                batch_size=4096,
                clip_grad_norm=1.0,
                eval_freq=5000,
                eval_batch_size=16, #500,
                device=device,
            ),
            data_path = '/mnt/nfs/chenxinyan/muzero/muzero_hidden_data/ver2_0430/data/latent_data_dict_2063.pth',
            data_path2 = '/mnt/nfs/chenxinyan/muzero/muzero_hidden_data/ver2_0430/data/latent_data_dict2_1090.pth',
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
