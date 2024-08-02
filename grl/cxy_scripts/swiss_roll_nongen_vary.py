################################################################################################
# This script demonstrates how to use an Independent Conditional Flow Matching (ICFM), which is a flow model, to train a world model by using Swiss Roll dataset.
################################################################################################

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
from torch.utils.data import Dataset
from easydict import EasyDict
from matplotlib import animation

from grl.datasets.swiss_roll_dataset import DynamicSwissRollDataset
from grl.generative_models.intrinsic_model import IntrinsicModel
from grl.generative_models.metric import compute_likelihood
from grl.utils import set_seed
from grl.utils.log import log

exp_name = "swiss-roll-nongen-varying-world-model-noise"

x_size = 2
condition_size=3
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
data_num=100000
config = EasyDict(
    dict(
        device=device,
        dataset=dict(
            varying_dynamic=True,
            n_samples=data_num,
            test_n_samples=data_num,
            pair_samples=10000,
            # delta_t_barrie=0.1,
            # noise=0.001,
            delta_t_barrie=0.2,
            noise=0.3,
        ),
        model=dict(
            t_encoder=dict(
                type="GaussianFourierProjectionTimeEncoder",
                args=dict(
                    embed_dim=t_embedding_dim,
                    scale=30.0,
                ),
            ),
            condition_encoder = dict(
                type="GaussianFourierProjectionEncoder",
                args=dict(
                    embed_dim=t_embedding_dim, # after flatten, 32*3=96
                    x_shape=(condition_size,),
                    scale=30.0,
                ),
            ),
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=x_size,
                    t_dim=t_embedding_dim,
                    condition_dim=t_embedding_dim*condition_size,
                    condition_hidden_dim=64,
                    t_condition_hidden_dim=128,
                ),
            ),
        ),
        parameter=dict(
            lr=5e-4,
            data_num=data_num,
            iterations=100000,
            batch_size=2000,
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
    nongen_model = IntrinsicModel(config.model).to(config.device)
    nongen_model = torch.compile(nongen_model)

    customized_dataset = DynamicSwissRollDataset(config.dataset, train=True)
    x=np.concatenate([customized_dataset.data['state'], customized_dataset.data['next_state']], axis=0)
    customized_dataset.plot(customized_dataset.data['state'], customized_dataset.data['next_state'], customized_dataset.data['action'], path=config.parameter.video_save_path, tag='dataset_plot')

    x0, x1, action, background = customized_dataset.data['state'], customized_dataset.data['next_state'], customized_dataset.data['action'], customized_dataset.data['background']
    #
    optimizer = torch.optim.Adam(
        nongen_model.parameters(),
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
            nongen_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    # zip x0, x1, action
    condition = torch.cat((torch.tensor(action).unsqueeze(1), torch.tensor(background)), dim=1).float()
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x0).to(config.device),
            torch.tensor(x1).to(config.device),
            torch.tensor(condition).to(config.device),
        ),
        batch_size=config.parameter.batch_size,
        shuffle=True,
    )
        

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    data_generator = get_train_data(data_loader)

    gradient_sum = 0.0
    loss_sum = 0.0
    counter = 0
    iteration = 0



    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
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

    def render_3d_trajectory_video(data, video_save_path, iteration, fps=100, dpi=100):

        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
            
        T, B, _ = data.shape
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set the axes limits
        ax.set_xlim(np.min(data[:,:,0]), np.max(data[:,:,0]))
        ax.set_ylim(np.min(data[:,:,1]), np.max(data[:,:,1]))
        ax.set_zlim(0, T)

        # Initialize a list of line objects for each point with alpha transparency
        lines = [ax.plot([], [], [], alpha=0.5)[0] for _ in range(B)]

        # Initialization function to set the background of each frame
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        # Animation function which updates each frame
        def update(frame):
            for i, line in enumerate(lines):
                x_data = data[:frame+1, i, 0]
                y_data = data[:frame+1, i, 1]
                z_data = np.arange(frame+1)
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            return lines

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=True)

        # Save the animation
        video_filename = os.path.join(video_save_path, f"iteration_3D_{iteration}.mp4")
        ani.save(video_filename, fps=fps, dpi=dpi)

        # Clean up
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
    

    save_checkpoint_on_exit(nongen_model, optimizer, history_iteration)

    for iteration in track(range(config.parameter.iterations), description="Training"):

        if iteration <= last_iteration:
            continue

        # if iteration > 0 and iteration % config.parameter.eval_freq == 0:
        # # if True:
        #     nongen_model.eval()
        #     t_span = torch.zeros((500,)).to(config.device)
        #     customized_eval_dataset = DynamicSwissRollDataset(config.dataset, train=True)
        #     x0_eval, x1_eval, action_eval, background_eval = customized_eval_dataset.data['state'], customized_eval_dataset.data['next_state'], customized_eval_dataset.data['action'], customized_eval_dataset.data['background']
        #     x0_eval = torch.tensor(x0_eval).to(config.device)
        #     x1_eval = torch.tensor(x1_eval).to(config.device)
        #     condition_eval = torch.cat((torch.tensor(action_eval).unsqueeze(1), torch.tensor(background_eval)), dim=1).float().to(config.device)

        #     # ramdom choose 500 samples from x0_eval, x1_eval, action_eval
        #     x0_eval = x0_eval[:500]
        #     x1_eval = x1_eval[:500]
        #     condition_eval = condition_eval[:500]
                                                      
        #     # action_eval = -torch.ones_like(action_eval).to(config.device)*0.05
        #     predict_eval = nongen_model(t=t_span, x=x0_eval, condition=condition_eval)
        #     # interpolate
        #     interpolated_list = np.linspace(x0_eval, predict_eval, 1000)  # uniform, (1000, num, dim)
        #     render_video(interpolated_list, config.parameter.video_save_path, iteration, fps=100, dpi=100)

        batch_data = next(data_generator)

        nongen_model.train()
        t_span = torch.zeros((batch_data[0].shape[0],)).to(config.device)
        x0_train, x1_train, condition_train = batch_data[0], batch_data[1], batch_data[2]
        predict_train = nongen_model(t=t_span, x=x0_train, condition=condition_train)
        optimizer.zero_grad()
        # loss: MSE
        loss = torch.nn.functional.mse_loss(predict_train, x1_train)
        loss.backward()
        gradien_norm = torch.nn.utils.clip_grad_norm_(
            nongen_model.parameters(), config.parameter.clip_grad_norm
        )
        optimizer.step()
        gradient_sum += gradien_norm.item()
        loss_sum += loss.item()
        counter += 1

        log.info(
            f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
        )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(nongen_model, optimizer, iteration)
