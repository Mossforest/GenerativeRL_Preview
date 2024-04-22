import os
from easydict import EasyDict
from rich.progress import track
import numpy as np
import h5py
import random
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
from torch.utils.data import random_split
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from generative_rl.machine_learning.generative_models.diffusion_model.diffusion_model import DiffusionModel

project="test_dynamic_v2_1_test"



class MetadriveDataset(Dataset):
    def __init__(self, data_list, key_list):
        self.key_list = key_list
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        return {key: item[idx] for key, item in zip(self.key_list, self.data_list)}


def get_dataset(data_path, train_ratio=0.9):

    data_dict = torch.load(data_path)
    data_list = list(data_dict.items())
    key_list = [item[0] for item in data_list]
    value_list = [item[1] for item in data_list]

    np.random.seed(0)  # 为了可重现性设置随机种子
    permuted_indices = np.random.permutation(len(value_list[0]))
    value_list = [item[permuted_indices] for item in value_list]

    train_size = int(len(value_list[0]) * train_ratio)
    train_list = [item[:train_size] for item in value_list]
    eval_list = [item[train_size:] for item in value_list]
    
    train_dataset = MetadriveDataset(train_list, key_list)
    eval_dataset = MetadriveDataset(eval_list, key_list)
    
    return train_dataset, eval_dataset


# def visual_data(data_list, target_data, video_save_path, iteration, pca=False):
#     if pca:
#         # PCA
#         scaler = StandardScaler()
#         scaler.fit(target_data)
#         X_scaled = scaler.transform(target_data)
#         # 创建PCA对象，选择主成分数量
#         pca = PCA(n_components=2)
#         pca.fit(X_scaled)
#         target_pca_data = pca.transform(X_scaled)

#         # 对标准化后的数据进行PCA处理
#         data_pca_list = []
#         for data in data_list:
#             data_scaled = scaler.transform(data)
#             data_pca = pca.transform(data_scaled)
#             data_pca_list.append(data_pca)
#     else:
#         # original data, visualize first 2 dim
#         target_pca_data = target_data[:, :2]
#         data_pca_list = []
#         for data in data_list:
#             data_pca_list.append(data[:, :2])
    
#     render_video(data_pca_list, target_pca_data, video_save_path, iteration, fps=100, dpi=100)



# def render_video(data_list, target_data, video_save_path, iteration, fps=100, dpi=100):
#     cmap_name = 'tab20'
#     color_num = target_data.shape[0]
#     # print('='*10, color_num)
#     # colors = cm.get_cmap(cmap_name, color_num)(range(color_num))
#     # print('='*10, colors)
    
#     colors = ['red', 'green', 'blue', 'grey', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
    
#     if not os.path.exists(video_save_path):
#         os.makedirs(video_save_path)
#     fig = plt.figure(figsize=(6, 6))
#     plt.xlim([-10, 10])
#     plt.ylim([-10, 10])
#     plt.scatter(target_data[:,0], target_data[:,1], s=1, c=colors)
#     # for d, c in zip(target_data, colors):
#     #     plt.scatter(d[0], d[1], s=1, c=c)
#     ims = []

#     for i, data in enumerate(data_list):
#         # image alpha frm 0 to 1
#         im = plt.scatter(data[:,0], data[:,1], s=1, c=colors)
#         # for d, c in zip(data, colors):
#         #     im = plt.scatter(d[:,0], d[:,1], s=1, c=c)
#         ims.append([im])
#     ani = animation.ArtistAnimation(fig, ims, interval=0.1, blit=True)
#     ani.save(os.path.join(video_save_path, f"iteration_{iteration}.mp4"), fps=fps, dpi=dpi)
#     # clean up
#     print(f'saved video iteration_{iteration}.mp4')
#     plt.close(fig)
#     plt.clf()


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
                                "score_function",
                            ],
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            parameters=dict(
                training_loss_type=dict(
                    values=["score_matching"],
                ),
                lr=dict(
                    values=[1e-4] # ,3e-3,4e-3,5e-3],
                ),
            ),
        ),
    ),
)


def main():
    action_size=4
    state_size=24
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # t_embedding_dim = 32
    # t_encoder = dict(
    #     type = "GaussianFourierProjectionTimeEncoder",
    #     args = dict(
    #         embed_dim = t_embedding_dim,
    #         scale = 30.0,
    #     ),
    # )
    config = EasyDict(
        dict(
            device = device,
            diffusion_model = dict(
                device = device,
                x_size = state_size,
                alpha = 1.0,
                solver = dict(
                    type = "ODESolver",
                    args = dict(
                        library="torchdyn",
                    ),
                ),
                path = dict(
                    type = "linear_vp_sde",
                    beta_0 = 0.1,
                    beta_1 = 20.0,
                ),
                model = dict(
                    type = "score_function",
                    args = dict(
                        # t_encoder = t_encoder,
                        backbone = dict(
                            type = "transformer_1d", #TODO
                            args = dict(
                                input_dim = 1,
                                sequence_dim = state_size,
                                hidden_dim = 128,
                                output_dim = 1,
                                condition_config = dict(
                                    backbone = dict(
                                        type = "ConcatenateMLP",
                                        args = dict(
                                            hidden_sizes = [action_size + state_size, 256, 256],
                                            output_size = 128,
                                            activation = "silu",
                                            layernorm = True,
                                            final_activation = "tanh",
                                            shrink = 0.1,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            parameter = dict(
                training_loss_type = "score_matching",
                lr=5e-4,
                weight_decay=0,
                iterations=100000,
                batch_size=4096,
                clip_grad_norm=1.0,
                eval_freq=5000,
                eval_batch_size=10, #500,
                device=device,
            ),
            data_path = '/mnt/nfs/chenxinyan/muzero/muzero_hidden_data/hidden/latent_data_dict.pth',
    ))

    with wandb.init(
            project=project,
            config=config,
    ) as wandb_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = EasyDict(wandb.config)
        run_name = 'no_weightdecay_flow'
        wandb.run.name = run_name
        wandb.run.save()
        
        
        # get data
        def get_train_data(dataloader):
            while True:
                yield from dataloader
        dataset, eval_dataset = get_dataset(config.train_data_path)
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
        
        for iteration in track(range(config.parameter.iterations), description="Training"):

            batch_data = next(data_generator)
            
            diffusion_model.train()
            if config.parameter.training_loss_type=="flow_matching":
                loss=diffusion_model.flow_matching_loss(x=batch_data['next_state'])
            elif config.parameter.training_loss_type=="score_matching":
                loss=diffusion_model.score_matching_loss(x=batch_data['next_state'])
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
                # TODO: waiting for GRL likelihood, FID, IS... metrics
                # eval_batch_data = next(eval_data_generator)
                
                # diffusion_model.eval()
                # t_span=torch.linspace(0.0, 1.0, 1000)
                # x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=config.parameter.eval_batch_size).cpu().detach()
                # x_t=[x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)]
                # eval_loss = torch.nn.functional.mse_loss(x_t[-1].squeeze(), eval_data_x)
                # print(f'eval_loss: {eval_loss.item()}')
                save_checkpoint(diffusion_model, optimizer, iteration, f'{project}/{run_name}/checkpoint')
                
                wandb_run.log(data=dict(iteration=iteration, gradient=gradient_sum / counter, loss=loss.item()), commit=True)
                # wandb_run.log(data=dict(iteration=iteration, gradient=gradient_sum / counter, loss=loss.item(), eval_loss=eval_loss.item()), commit=True)
            else:
                wandb_run.log(data=dict(iteration=iteration, gradient=gradient_sum / counter, loss=loss.item()), commit=True)

if __name__ == '__main__':
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=project
    )
    wandb.agent(sweep_id, function=main)
