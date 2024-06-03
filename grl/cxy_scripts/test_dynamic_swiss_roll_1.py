import multiprocessing as mp
import os
import signal
import sys
import wandb
import datetime

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import Progress, track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from easydict import EasyDict
from matplotlib import animation

from grl.generative_models.diffusion_model.dynamic_diffusion_model import DynamicDiffusionModel
from grl.cxy_models.classifier import Classifier
from grl.utils import set_seed
from grl.utils.log import log

x_size = 2

project = "test_dynamic_swiss_roll"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
data_num = 10000
config = EasyDict(
    project=project,
    dataset=dict(
        data_num=data_num,
        noise=0.6,
        temperature=0.1,
    ),
    model=dict(
        device=device,
        value_function_model=dict(
            device=device,
            v_alpha=1.0,
            DoubleVNetwork=dict(
                state_encoder=dict(
                    type="GaussianFourierProjectionEncoder",
                    args=dict(
                        embed_dim=128,
                        x_shape=[x_size],
                        scale=0.5,
                    ),
                ),
                backbone=dict(
                    type="DynamicDiT_1D",
                    args=dict(
                        hidden_sizes=[128 * x_size, 256, 256],
                        output_size=1,
                        activation="silu",
                    ),
                ),
            ),
        ),
        diffusion_model=dict(
            device=device,
            x_size=x_size,
            alpha=1.0,
            solver=dict(
                type="SDESolver",
                args=dict(
                    library="torchdyn",
                        sde_solver="euler_script",
                ),
            ),
            path=dict(
                type="gvp", # linear_vp_sde
                beta_0=0.1,
                beta_1=20.0,
            ),
            model=dict(
                type="score_script_function",
                args=dict(
                    t_encoder=t_encoder,
                    backbone=dict(
                        type="TemporalSpatialResidualNet",
                        args=dict(
                            hidden_sizes=[512, 256, 128],
                            output_dim=x_size,
                            t_dim=t_embedding_dim,
                        ),
                    ),
                ),
            ),
        ),
    ),
    parameter=dict(
        training_loss_type = "score_matching",
        batch_size=2048,
        learning_rate=5e-5,
        iterations=50000,
        support_size=data_num,
        sample_per_data=100,
        discount_factor=0.99,
        discount_factor=0.99,
        update_momentum=0.995,
        evaluation=dict(
            eval_freq=5000,
            video_save_path="./video-swiss-roll-energy-conditioned-diffusion-model",
            model_save_path="./model-swiss-roll-energy-conditioned-diffusion-model",
            guidance_scale=[0, 1, 2, 4, 8, 16],
        ),
    ),
)


def render_video(
    data_list, video_save_path, iteration, guidance_scale=1.0, fps=100, dpi=100
):
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
        os.path.join(
            video_save_path,
            f"iteration_{iteration}_guidance_scale_{guidance_scale}.mp4",
        ),
        fps=fps,
        dpi=dpi,
    )
    # clean up
    plt.close(fig)
    plt.clf()


def save_checkpoint(model, iteration, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            model=model.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"checkpoint_{iteration}.pt"),
    )


def save_checkpoint(
    diffusion_model, value_model, diffusion_model_iteration, value_model_iteration, path
):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            diffusion_model=diffusion_model.state_dict(),
            value_model=value_model.state_dict(),
            diffusion_model_iteration=diffusion_model_iteration,
            value_model_iteration=value_model_iteration,
        ),
        f=os.path.join(
            path, f"checkpoint_{diffusion_model_iteration+value_model_iteration}.pt"
        ),
    )


def main(wandb_run):
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    
    # ! get data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # get data
    x_and_t = make_swiss_roll(
        n_samples=config.dataset.data_num, noise=config.dataset.noise
    )
    t = x_and_t[1].astype(np.float32)
    t = (t - np.min(t)) / (np.max(t) - np.min(t))   # [0, 1]
    x = x_and_t[0].astype(np.float32)[:, [0, 2]]
    # transform data
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 10 - 5

    # plot data with color of value
    plt.scatter(x[:, 0], x[:, 1], c=t, vmin=-5, vmax=3)
    plt.colorbar()
    if not os.path.exists(config.parameter.evaluation.video_save_path):
        os.makedirs(config.parameter.evaluation.video_save_path)
    plt.savefig(
        os.path.join(
            config.parameter.evaluation.video_save_path, f"swiss_roll_data_origin.png"
        )
    )
    plt.clf()
    
    # pair the sampled (x, t) 2by2
    pair_sampled_num = int(1e5)
    delta_t_barrie = 0.15
    
    idx_1 = torch.randint(config.dataset.data_num, (pair_sampled_num,))
    idx_2 = torch.randint(config.dataset.data_num, (pair_sampled_num,))
    unfil_x_1 = x[idx_1]
    unfil_t_1 = t[idx_1]
    unfil_x_2 = x[idx_2]
    unfil_t_2 = t[idx_2]
    unfil_delta_t = unfil_t_1 - unfil_t_2
    
    idx_fil = (unfil_delta_t > - delta_t_barrie) & (unfil_delta_t < delta_t_barrie)
    x_1 = unfil_x_1[idx_fil]
    x_2 = unfil_x_2[idx_fil]
    delta_t = unfil_delta_t[idx_fil]
    
    # TODO: distributed stochastic rule
    
    data = np.concatenate([x_1, x_2, delta_t[:, None]], axis=1)
    # array([ 1.7171526 , -0.5514145 ,  1.5585546 ,  0.50061655, -0.06819396], dtype=float32)  ==  [x1, x2, dt]
    
    
    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.parameter.unconditional_model.batch_size, shuffle=True
    )
    data_generator = get_train_data(data_loader)
    
    
    
    # ! build model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    diffusion_model = DynamicDiffusionModel(config=config.diffusion_model).to(config.diffusion_model.device)
    wandb_run.watch(diffusion_model, log='all')

    if config.parameter.evaluation.model_save_path is not None:
        if not os.path.exists(config.parameter.evaluation.model_save_path):
            log.warning(
                f"Checkpoint path {config.parameter.evaluation.model_save_path} does not exist"
            )
            diffusion_model_iteration = 0
            value_model_iteration = 0
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.evaluation.model_save_path)
                if f.endswith(".pt")
            ]
            if len(checkpoint_files) == 0:
                log.warning(
                    f"No checkpoint files found in {config.parameter.evaluation.model_save_path}"
                )
                diffusion_model_iteration = 0
                value_model_iteration = 0
            else:
                checkpoint_files = sorted(
                    checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
                checkpoint = torch.load(
                    os.path.join(
                        config.parameter.evaluation.model_save_path,
                        checkpoint_files[-1],
                    ),
                    map_location="cpu",
                )
                diffusion_model.load_state_dict(checkpoint["diffusion_model"])
                diffusion_model_iteration = checkpoint.get("diffusion_model_iteration", 0)

    else:
        diffusion_model_iteration = 0
        value_model_iteration = 0
    
    # optim
    diffusion_optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.unconditional_model.learning_rate,
    )
    


    moving_average_loss = 0.0

    subprocess_list = []

    for train_iter in track(
        range(config.parameter.unconditional_model.iterations),
        description="unconditional_model training",
    ):
        if train_iter < diffusion_model_iteration:
            continue

        train_data = next(data_generator).to(config.model.diffusion_model.device)
        train_x, train_value = train_data[:, :x_size], train_data[:, x_size]
        diffusion_training_loss = (
            diffusion_model.score_matching_loss(train_x)
        )
        diffusion_optimizer.zero_grad()
        diffusion_training_loss.backward()
        moving_average_loss = (
            0.99 * moving_average_loss + 0.01 * diffusion_training_loss.item()
            if train_iter > 0
            else diffusion_training_loss.item()
        )
        if train_iter % 100 == 0:
            log.info(
                f"iteration {train_iter}, unconditional model loss {diffusion_training_loss.item()}, moving average loss {moving_average_loss}"
            )

        diffusion_model_iteration = train_iter

        if (
            train_iter == 0
            or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
        ):
            for guidance_scale in [0.0]:
                t_span = torch.linspace(0.0, 1.0, 1000)
                x_t = (
                    diffusion_model.sample_forward_process(
                        t_span=t_span, batch_size=500, guidance_scale=guidance_scale
                    )
                    .cpu()
                    .detach()
                )
                x_t = [
                    x.squeeze(0)
                    for x in torch.split(x_t, split_size_or_sections=1, dim=0)
                ]
                p = mp.Process(
                    target=render_video,
                    args=(
                        x_t,
                        config.parameter.evaluation.video_save_path,
                        f"diffusion_model_iteration_{diffusion_model_iteration}_value_model_iteration_{value_model_iteration}",
                        guidance_scale,
                        100,
                        100,
                    ),
                )
                p.start()
                subprocess_list.append(p)
            save_checkpoint(
                diffusion_model=diffusion_model,
                diffusion_model_iteration=diffusion_model_iteration,
                path=config.parameter.evaluation.model_save_path,
            )

    for p in subprocess_list:
        p.join()

    # def generate_fake_x(model, sample_per_data):
    #     # model.eval()
    #     fake_x_sampled = []
    #     for i in track(
    #         range(config.parameter.support_size), description="Generate fake x"
    #     ):
    #         # TODO: mkae it batchsize

    #         fake_x_sampled.append(
    #             model.sample(
    #                 t_span=torch.linspace(0.0, 1.0, 32).to(
    #                     config.model.diffusion_model.device
    #                 ),
    #                 batch_size=sample_per_data,
    #                 guidance_scale=0.0,
    #                 with_grad=False,
    #             )
    #         )

    #     fake_x = torch.stack(fake_x_sampled, dim=0)
    #     return fake_x

    # fake_x = generate_fake_x(
    #     diffusion_model, config.parameter.sample_per_data
    # )

    # # fake_x
    # data_fake_x = fake_x.detach().cpu().numpy()

    # data_loader = torch.utils.data.DataLoader(
    #     data, batch_size=config.parameter.value_function_model.batch_size, shuffle=True
    # )
    # data_loader_fake_x = torch.utils.data.DataLoader(
    #     data_fake_x,
    #     batch_size=config.parameter.energy_guidance.batch_size,
    #     shuffle=True,
    # )
    # data_generator = get_train_data(data_loader)
    # data_generator_fake_x = get_train_data(data_loader_fake_x)


    # energy_guidance_optimizer = torch.optim.Adam(
    #     energy_conditioned_diffusion_model.energy_guidance.parameters(),
    #     lr=config.parameter.energy_guidance.learning_rate,
    # )

    # moving_average_v_loss = 0.0
    # moving_average_energy_guidance_loss = 0.0

    # subprocess_list = []

    # with Progress() as progress:
    #     value_training = progress.add_task(
    #         "Value training",
    #         total=config.parameter.value_function_model.stop_training_iterations,
    #     )
    #     energy_guidance_training = progress.add_task(
    #         "Energy guidance training",
    #         total=config.parameter.energy_guidance.iterations,
    #     )

    #     for train_iter in range(config.parameter.energy_guidance.iterations):

    #         if train_iter < value_model_iteration:
    #             continue

    #         if train_iter % config.parameter.evaluation.eval_freq == 0:
    #             # mesh grid from -10 to 10
    #             x = np.linspace(-10, 10, 100)
    #             y = np.linspace(-10, 10, 100)
    #             grid = np.meshgrid(x, y)
    #             grid = np.stack([grid[1], grid[0]], axis=0)
    #             grid_tensor = torch.tensor(grid, dtype=torch.float32).to(
    #                 config.model.diffusion_model.device
    #             )
    #             grid_tensor = torch.einsum("dij->ijd", grid_tensor)

    #             # plot value function by imshow
    #             grid_value = value_function_model(grid_tensor)
    #             # plt.imshow(torch.fliplr(grid_value).detach().cpu().numpy(), extent=(-10, 10, -10, 10))
    #             plt.imshow(
    #                 grid_value.detach().cpu().numpy(),
    #                 extent=(-10, 10, -10, 10),
    #                 vmin=-5,
    #                 vmax=3,
    #             )
    #             plt.colorbar()
    #             if not os.path.exists(config.parameter.evaluation.video_save_path):
    #                 os.makedirs(config.parameter.evaluation.video_save_path)
    #             plt.savefig(
    #                 os.path.join(
    #                     config.parameter.evaluation.video_save_path,
    #                     f"iteration_{train_iter}_value_function.png",
    #                 )
    #             )
    #             plt.clf()

    #         train_data = next(data_generator).to(config.model.diffusion_model.device)
    #         train_x, train_value = train_data[:, :x_size], train_data[:, x_size]
    #         train_fake_x = next(data_generator_fake_x).to(
    #             config.model.diffusion_model.device
    #         )
    #         if (
    #             train_iter
    #             < config.parameter.value_function_model.stop_training_iterations
    #         ):
    #             v_loss = value_function_model.v_loss(
    #                 state=train_x,
    #                 value=train_value.unsqueeze(-1),
    #             )

    #             v_optimizer.zero_grad()
    #             v_loss.backward()
    #             v_optimizer.step()
    #             moving_average_v_loss = (
    #                 0.99 * moving_average_v_loss + 0.01 * v_loss.item()
    #                 if train_iter > 0
    #                 else v_loss.item()
    #             )
    #             if train_iter % 100 == 0:
    #                 log.info(
    #                     f"iteration {train_iter}, value loss {v_loss.item()}, moving average loss {moving_average_v_loss}"
    #                 )

    #             # Update target
    #             for param, target_param in zip(
    #                 value_function_model.v.parameters(),
    #                 value_function_model.v_target.parameters(),
    #             ):
    #                 target_param.data.copy_(
    #                     config.parameter.value_function_model.update_momentum
    #                     * param.data
    #                     + (1 - config.parameter.value_function_model.update_momentum)
    #                     * target_param.data
    #                 )

    #             progress.update(value_training, advance=1)

    #         energy_guidance_loss = (
    #             energy_conditioned_diffusion_model.energy_guidance_loss(
    #                 x=train_fake_x,
    #             )
    #         )
    #         energy_guidance_optimizer.zero_grad()
    #         energy_guidance_loss.backward()
    #         energy_guidance_optimizer.step()
    #         moving_average_energy_guidance_loss = (
    #             0.99 * moving_average_energy_guidance_loss
    #             + 0.01 * energy_guidance_loss.item()
    #             if train_iter > 0
    #             else energy_guidance_loss.item()
    #         )
    #         if train_iter % 100 == 0:
    #             log.info(
    #                 f"iteration {train_iter}, energy guidance loss {energy_guidance_loss.item()}, moving average loss {moving_average_energy_guidance_loss}"
    #             )

    #         value_model_iteration = train_iter
    #         progress.update(energy_guidance_training, advance=1)

    #         if (
    #             train_iter == 0
    #             or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
    #         ):
    #             energy_conditioned_diffusion_model.eval()
    #             for guidance_scale in config.parameter.evaluation.guidance_scale:
    #                 t_span = torch.linspace(0.0, 1.0, 1000)
    #                 x_t = (
    #                     energy_conditioned_diffusion_model.sample_forward_process(
    #                         t_span=t_span, batch_size=500, guidance_scale=guidance_scale
    #                     )
    #                     .cpu()
    #                     .detach()
    #                 )
    #                 x_t = [
    #                     x.squeeze(0)
    #                     for x in torch.split(x_t, split_size_or_sections=1, dim=0)
    #                 ]
    #                 p = mp.Process(
    #                     target=render_video,
    #                     args=(
    #                         x_t,
    #                         config.parameter.evaluation.video_save_path,
    #                         f"diffusion_model_iteration_{diffusion_model_iteration}_value_model_iteration_{value_model_iteration}",
    #                         guidance_scale,
    #                         100,
    #                         100,
    #                     ),
    #                 )
    #                 p.start()
    #                 subprocess_list.append(p)

    #             save_checkpoint(
    #                 diffusion_model=energy_conditioned_diffusion_model,
    #                 value_model=value_function_model,
    #                 diffusion_model_iteration=diffusion_model_iteration,
    #                 value_model_iteration=value_model_iteration,
    #                 path=config.parameter.evaluation.model_save_path,
    #             )

    # for p in subprocess_list:
    #     p.join()


if __name__ == '__main__':
    with wandb.init(
            project="test_swiss_roll_1",
            config=config,
    ) as wandb_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = EasyDict(wandb.config)
        wandb.run.name = project
        wandb.run.save()
        
        main(wandb_run)