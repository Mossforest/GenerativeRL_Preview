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
from grl.cxy_models.swiss_roll_dataset import DynamicSwissRollDataset

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
        n_samples=data_num,
        pair_samples=1e5,
        delta_t_barrie=0.15,
        noise=0.6,
        temperature=0.1,
        test_n_samples=data_num*0.3,
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
        classifier=dict(
            stata_size=123,
            action_size=123,
            background_size=123,
            hiddensize=123,
        ),
    ),
    parameter=dict(
        training_loss_type = "score_matching",
        batch_size=2048,
        learning_rate=5e-5,
        classifier_alpha=0.3,
        iterations=50000,
        support_size=data_num,
        sample_per_data=100,
        discount_factor=0.99,
        discount_factor=0.99,
        update_momentum=0.995,
        evaluation=dict(
            eval_freq=5000,
            batch_size=1000,
            video_save_path="./video-swiss-roll-energy-conditioned-diffusion-model",
            model_save_path="./model-swiss-roll-energy-conditioned-diffusion-model",
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
    diffusion_model, classifier, diffusion_model_iteration, classifier_iteration, path
):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            diffusion_model=diffusion_model.state_dict(),
            classifier=classifier.state_dict(),
            diffusion_model_iteration=diffusion_model_iteration,
            classifier_iteration=classifier_iteration,
        ),
        f=os.path.join(
            path, f"checkpoint_{diffusion_model_iteration+classifier_iteration}.pt"
        ),
    )


def main(wandb_run):
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    
    # ! get data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # get data
    def get_train_data(dataloader):
        while True:
            yield from dataloader
    dataset = DynamicSwissRollDataset(config, train=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.parameter.batch_size, shuffle=True
    )
    data_generator = get_train_data(data_loader)
    
    test_dataset = DynamicSwissRollDataset(config, train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.parameter.evaluation.batch_size, shuffle=True
    )
    test_data_generator = get_train_data(test_data_loader)
    eval_data = next(test_data_generator).to(config.model.diffusion_model.device)
    
    
    # ! build model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    diffusion_model = DynamicDiffusionModel(config=config.diffusion_model).to(config.diffusion_model.device)
    wandb_run.watch(diffusion_model, log='all')
    
    classifier = Classifier(config=config.classifier).to(config.diffusion_model.device)
    wandb_run.watch(classifier, log='all')

    if config.parameter.evaluation.model_save_path is not None:
        if not os.path.exists(config.parameter.evaluation.model_save_path):
            log.warning(
                f"Checkpoint path {config.parameter.evaluation.model_save_path} does not exist"
            )
            diffusion_model_iteration = 0
            classifier_iteration = 0
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
                classifier_iteration = 0
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
                # TODO: load classifier

    else:
        diffusion_model_iteration = 0
        classifier_iteration = 0
    
    # optim
    parameters = list(diffusion_model.model.parameters() + classifier.parameters())
    diffusion_optimizer = torch.optim.Adam(
        parameters,
        lr=config.parameter.learning_rate,
    )

    moving_average_loss = 0.0


    # ! training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for train_iter in track(
        range(config.parameter.iterations),
        description="stage3 training",
    ):
        if train_iter < diffusion_model_iteration:
            continue

        train_data = next(data_generator).to(config.model.diffusion_model.device)
        train_x, train_a, train_bg, train_next_x = train_data['state'], train_data['action'], train_data['background'], train_data['next_x']
        train_condition = {'action': train_a, 'dynamic': train_bg}
        classifier_weight = classifier(train_x, train_a, train_bg)
        classifier_loss = classifier.loss(
            classifier_weight, 
            train_x, 
            train_next_x
        )

        diffusion_loss = diffusion_model.score_matching_loss(
            train_next_x, 
            train_x, 
            train_condition,
            classifier_weight
        )
        diffusion_training_loss = diffusion_loss + config.parameter.classifier_alpha * classifier_loss
        diffusion_optimizer.zero_grad()
        diffusion_training_loss.backward()
        moving_average_loss = (
            0.99 * moving_average_loss + 0.01 * diffusion_training_loss.item()
            if train_iter > 0
            else diffusion_training_loss.item()
        )
        if train_iter % 100 == 0:
            log.info(
                f"iteration {train_iter}, diffusion model loss {diffusion_loss.item():6f}, classifier loss {classifier_loss.item():6f}, training loss {diffusion_training_loss.item():6f}, moving average loss {moving_average_loss:6f}"
            )

        diffusion_model_iteration = train_iter

        # ! evaluate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (
            train_iter == 0
            or (train_iter + 1) % config.parameter.evaluation.eval_freq == 0
        ):
            with torch.no_grad():
                eval_x, eval_a, eval_bg, eval_next_x = eval_data['state'], eval_data['action'], eval_data['background'], eval_data['next_x']
                eval_condition = {'action': eval_a, 'dynamic': eval_bg}
                
                classifier_weight = classifier(eval_x, eval_a, eval_bg)
                classifier_loss = classifier.loss(classifier_weight, eval_x, eval_next_x)
                
                t_span = torch.linspace(0.0, 1.0, 1000)
                x_t = (
                    diffusion_model.sample_forward_process(
                        eval_x, eval_condition, t_span=t_span, batch_size=500, with_grad=False
                    )
                    .cpu()
                    .detach()
                )
                x_t = [x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)]
                render_video(
                        x_t,
                        config.parameter.evaluation.video_save_path,
                        f"diffusion_model_iteration_{diffusion_model_iteration}_classifier_iteration_{classifier_iteration}",
                        100,
                        100,
                )
                save_checkpoint(
                    diffusion_model=diffusion_model,
                    diffusion_model_iteration=diffusion_model_iteration,
                    classifier=classifier,
                    classifier_iteration=classifier_iteration,
                    path=config.parameter.evaluation.model_save_path,
                )


if __name__ == '__main__':
    with wandb.init(
            project="test_swiss_roll_3",
            config=config,
    ) as wandb_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = EasyDict(wandb.config)
        wandb.run.name = project
        wandb.run.save()
        
        main(wandb_run)