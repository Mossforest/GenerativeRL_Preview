################################################################################################
# This script demonstrates how to use an Independent Conditional Flow Matching (ICFM), which is a flow model, to train a world model by using Swiss Roll dataset.
################################################################################################

import os
import signal
import sys

import wandb
import matplotlib
import numpy as np
from easydict import EasyDict
from tensordict import TensorDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from matplotlib import animation

from grl.datasets.gp import GPD4RLTensorDictDataset
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
from grl.generative_models.metric import compute_likelihood
from grl.utils import set_seed
from grl.utils.log import log

proj_name = "d4rl-hopper"
sub_name = "medium-gnn"
exp_name = f"{proj_name}-{sub_name}"

env_id='hopper-medium-v2'
x_size = 11
action_size=3  # include action and background
background_size = 1 # fixed value
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
condition_dim=256
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
condition_encoder = dict(
    type="MultiMLPEncoder",
    args=dict(
        action_encoder=dict(
            hidden_sizes=[action_size] + [condition_dim] * 2,
            output_size=condition_dim,
            activation='relu',
        ),
        background_encoder=dict(
            hidden_sizes=[background_size] + [condition_dim] * 2,
            output_size=condition_dim,
            activation='relu',
        )
    ),
)
x_encoder = dict(
    type="MLPEncoder",
    args=dict(
        hidden_sizes=[x_size] + [condition_dim] * 2,
        output_size=condition_dim,
        activation='relu',
    ),
)
data_num=100000
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
                sigma=0.1,
            ),
            model=dict(
                type="velocity_function",
                args=dict(
                    t_encoder=t_encoder,
                    condition_encoder=condition_encoder,
                    x_encoder=x_encoder,
                    backbone=dict(
                        type="HeterogeneousGraphModel",
                        args=dict(
                            hidden_sizes=[512, 256, 128],
                            output_dim=x_size,
                            device=device,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=5e-4,
            data_num=data_num,
            num_iterations=100,
            batch_size=2048,
            clip_grad_norm=1.0,
            eval_freq=2000,
            checkpoint_freq=2000,
            checkpoint_path=f"./{exp_name}/checkpoint",
            video_save_path=f"./{exp_name}/video",
            device=device,
        ),
    )
)


def plot2d(data):

    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    wandb.init(project=proj_name, name=sub_name)
    wandb.config.update(config)
    
    flow_model = IndependentConditionalFlowModel(config=config.flow_model).to(
        config.flow_model.device
    )
    flow_model = torch.compile(flow_model)
    
    
    dataset = GPD4RLTensorDictDataset(env_id=env_id)

    replay_buffer=TensorDictReplayBuffer(
        storage=dataset.storage,
        batch_size=config.parameter.batch_size,
        sampler=SamplerWithoutReplacement(),
        prefetch=10,
        pin_memory=True,
    )
    
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

    def save_checkpoint_on_exit(model, optimizer, iterations):
        def exit_handler(signal, frame):
            log.info("Saving checkpoint when exit...")
            save_checkpoint(model, optimizer, iteration=iterations[-1])
            log.info("Done.")
            sys.exit(0)

        signal.signal(signal.SIGINT, exit_handler)


    gradient_sum = 0.0
    loss_sum = 0.0
    counter = 0
    history_iteration = [-1]
    save_checkpoint_on_exit(flow_model, optimizer, history_iteration)


    for iteration in track(range(config.parameter.num_iterations), description="Training"):

        for index, batch_data in enumerate(replay_buffer):
            
            action=batch_data["a"].to(config.device).to(torch.float32)
            state=batch_data["s"].to(config.device).to(torch.float32)
            next_state=batch_data["s_"].to(config.device).to(torch.float32)
            condition = TensorDict()
            condition['action'] = action
            condition['background'] = torch.zeros((action.shape[0], 1)).to(config.device).to(torch.float32)

            flow_model.train()
            loss = flow_model.flow_matching_loss(x0=state, x1=next_state, condition=condition)
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
            
            wandb.log({
                'iteration': iteration,
                'step': counter,
                'loss': loss.item(),
            }, step=counter)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(flow_model, optimizer, iteration)
