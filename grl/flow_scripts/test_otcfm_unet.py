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
                    t_encoder=t_encoder,
                    condition_encoder=dict(
                        type="GaussianFourierProjectionEncoder",
                        args=dict(
                            embed_dim=32,
                            x_shape=[1],
                            scale=0.5,
                        ),
                    ),
                    backbone=dict(
                        type="TemporalSpatialResidualNet",
                        args=dict(
                            hidden_sizes=[512, 256, 128],
                            output_dim=x_size,
                            t_dim=t_embedding_dim,
                            condition_dim=32,
                            condition_hidden_dim=64,
                            t_condition_hidden_dim=128,
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
                video_save_path="./video-swiss-roll-otcfm-unet",
                model_save_path="./model-swiss-roll-otcfm-unet",
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
