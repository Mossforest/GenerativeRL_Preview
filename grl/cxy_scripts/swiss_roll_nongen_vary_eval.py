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

exp_name = "swiss-roll-nongen-varying-world-model-mlpencoder"

x_size = 2
condition_size=3
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
condition_dim=256
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
                type="MLPEncoder",
                args=dict(
                    hidden_sizes=[condition_size] + [condition_dim] * 2,
                    output_size=condition_dim,
                    activation='relu',
                ),
            ),
            backbone=dict(
                type="TemporalSpatialResidualNet",
                args=dict(
                    hidden_sizes=[512, 256, 128],
                    output_dim=x_size,
                    t_dim=t_embedding_dim,
                    condition_dim=condition_dim,
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

    assert config.parameter.checkpoint_path is not None

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
    
    def render_eval_video(origin_line, data_list, video_save_path, video_name, title, fps=100, dpi=100):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        ims = []
        colors = np.linspace(0, 1, len(data_list))
        
        origin_line_plot = plt.scatter(origin_line[:, 0], origin_line[:, 1], s=1, alpha=0.8, c='yellow', label='Origin Line')
        plt.title(title)

        for i, data in enumerate(data_list):
            # image alpha frm 0 to 1
            im = plt.scatter(data[:, 0], data[:, 1], s=1, c='red')
            ims.append([im])
        ims = [im + [origin_line_plot] for im in ims]
        
        ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
        ani.save(
            os.path.join(video_save_path, f"{video_name}.mp4"),
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

    def save_checkpoint_on_exit(model, optimizer, iterations):
        def exit_handler(signal, frame):
            log.info("Saving checkpoint when exit...")
            save_checkpoint(model, optimizer, iteration=iterations[-1])
            log.info("Done.")
            sys.exit(0)

        signal.signal(signal.SIGINT, exit_handler)
    
    nongen_model.eval()
    t_span = torch.zeros((500,)).to(config.device)
    param_list = [1.5, 1.75, 2, 1.25, 1, 1.33, 1.84, 2.5, 0.75]
    
    for param in param_list:
        customized_eval_dataset = DynamicSwissRollDataset(config.dataset, train=True, varying_param_list=[param])
        x0_eval, x1_eval, action_eval, background_eval = customized_eval_dataset.data['state'], customized_eval_dataset.data['next_state'], customized_eval_dataset.data['action'], customized_eval_dataset.data['background']
        x0_eval = torch.tensor(x0_eval).to(config.device)
        x1_eval = torch.tensor(x1_eval).to(config.device)
        condition_eval = torch.cat((torch.tensor(action_eval).unsqueeze(1), torch.tensor(background_eval)), dim=1).float().to(config.device)
        action_eval = torch.tensor(action_eval).to(config.device)

        # ramdom choose 500 samples from x0_eval, x1_eval, action_eval
        x0_eval = x0_eval[:500]
        x1_eval = x1_eval[:500]
        condition_eval = condition_eval[:500]
        action_eval = action_eval[:500]
        
        # customized_eval_dataset.plot(x0_eval, x1_eval, action_eval, path=config.parameter.video_save_path, tag='original_roll')
        
        origin_line = customized_eval_dataset.get_origin_line(500)[0] # param_0
        
        predict_eval = nongen_model(t=t_span, x=x0_eval, condition=condition_eval)
        loss = torch.nn.functional.mse_loss(predict_eval, x1_eval)
        # interpolate
        interpolated_list = np.linspace(x0_eval.cpu(), predict_eval.detach().cpu(), 1000)  # uniform, (1000, num, dim)
        render_eval_video(x1_eval.cpu(), interpolated_list, config.parameter.video_save_path, f"eval_video_param_{param}", f'param: {param}, loss: {loss}', fps=100, dpi=100)
