import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll


class DynamicSwissRollDataset(Dataset):
    def __init__(self, config, train=True, varying_param_list=[1.5, 1.75, 2, 1.25, 1]):
        self.config = config
        self.train = train
        self.varying = config.varying_dynamic
        
        self.n_samples = int(config.n_samples) if train else int(config.test_n_samples)
        self.pair_samples = int(config.pair_samples) if train else int(config.pair_samples * config.test_n_samples / config.n_samples)
        self.delta_t_barrie = config.delta_t_barrie
        self.noise = config.noise
        self.varying_param_list = varying_param_list
        self.origin_line_list = []
        
        if not self.varying:
            x_1, x_2, action, bg, origin_line = self.make_dynamic_swiss_roll(self.n_samples, noise=self.noise, a=1.5, b=2.)
            self.origin_line_list.append(origin_line)
        
            self.data = {
                'state': x_1,
                'next_state': x_2,
                'action': action,
                'background': bg,
                }
        else:
            self.data = {
                'state': [],
                'next_state': [],
                'action': [],
                'background': [],
                }
            alist = np.array(self.varying_param_list)
            blist = 3 / alist
            for a, b in zip(alist, blist):
                x_1, x_2, action, bg, origin_line = self.make_dynamic_swiss_roll(self.n_samples, noise=self.noise, a=a, b=b)
                self.data['state'].append(x_1)
                self.data['next_state'].append(x_2)
                self.data['action'].append(action)
                self.data['background'].append(bg)
                self.origin_line_list.append(origin_line)
            
            self.data['state'] = np.concatenate(self.data['state'], axis=0)
            self.data['next_state'] = np.concatenate(self.data['next_state'], axis=0)
            self.data['action'] = np.concatenate(self.data['action'], axis=0)
            self.data['background'] = np.concatenate(self.data['background'], axis=0)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}
    
    def make_swiss_roll(self, n_samples=100, noise=0.0, a = 1.5, b = 1):
        generator = np.random.mtrand._rand
        t = a * np.pi * (b + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)

        x = t * np.cos(t)
        z = t * np.sin(t)

        origin_X = np.vstack((x, y, z))
        noisee = noise * generator.standard_normal(size=(3, n_samples))
        X = origin_X + noisee
        X = X.T
        origin_X = origin_X.T
        noisee = noisee.T
        t = np.squeeze(t)

        return X, t, noisee, origin_X
    
    def make_dynamic_swiss_roll(self, n_samples, noise, a, b):
        X, t, noise_gap, origin_line = self.make_swiss_roll(
            n_samples=n_samples, noise=noise, a=a, b=b
        )
        t = t.astype(np.float32)
        t = (t - np.min(t)) / (np.max(t) - np.min(t))   # [0, 1]
        x = X.astype(np.float32)[:, [0, 2]]
        origin_line = origin_line.astype(np.float32)[:, [0, 2]]
        noise_gap = noise_gap.astype(np.float32)[:, [0, 2]]
        noise_gap = np.linalg.norm(noise_gap, axis=1)
        
        # transform data
        x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
        x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 10 - 5
        origin_line[:, 0] = origin_line[:, 0] / np.max(np.abs(origin_line[:, 0]))
        origin_line[:, 1] = origin_line[:, 1] / np.max(np.abs(origin_line[:, 1]))
        origin_line = (origin_line - origin_line.min()) / (origin_line.max() - origin_line.min())
        origin_line = origin_line * 10 - 5
        
        # pair the sampled (x, t) 2by2
        idx_1 = torch.randint(n_samples, (self.pair_samples,))
        idx_2 = torch.randint(n_samples, (self.pair_samples,))
        unfil_x_1 = x[idx_1]
        unfil_t_1 = t[idx_1]
        unfil_x_2 = x[idx_2]
        unfil_t_2 = t[idx_2]
        noise_gap = noise_gap[idx_2]  # next_state
        unfil_delta_t = unfil_t_1 - unfil_t_2
        
        # rule 1: action - delta_t < delta_t_barrie
        idx_fil = np.abs(unfil_delta_t) < self.delta_t_barrie
        x_1 = unfil_x_1[idx_fil]
        x_2 = unfil_x_2[idx_fil]
        delta_t = unfil_delta_t[idx_fil]
        noise_gap = noise_gap[idx_fil]
        
        # rule 2: distributed stochastic rule
        idx_fil_2 = (np.abs(delta_t) < 0.5 * self.delta_t_barrie) | (noise_gap < 0.5 * noise)
        x_1 = x_1[idx_fil_2]
        x_2 = x_2[idx_fil_2]
        delta_t = delta_t[idx_fil_2]
        
        self.data_num = x_1.shape[0]
        
        bg = np.repeat(np.array([[a, b]]), self.data_num, axis=0)
        
        return (x_1, x_2, delta_t, bg, origin_line)
    
    def get_origin_line(self, n_samples=100):
        return self.origin_line_list
    
    def plot(self, x0, x1, t, path, tag=''):
        if type(x0) == torch.Tensor:
            x0 = x0.cpu()
            x1 = x1.cpu()
            t = t.cpu()
        
        # plot data with color of value
        plt.scatter(x0[:, 0], x0[:, 1], c=t, vmin=-0.1, vmax=0.1)
        plt.scatter(x1[:, 0], x1[:, 1], c=t, vmin=-0.1, vmax=0.1, marker="x")
        # plot line that connects x0 and x1
        plt.plot([x0[:, 0], x1[:, 0]], [x0[:, 1], x1[:, 1]], c="black", alpha=0.1)
        plt.colorbar()
        plt.title(tag)
        plt.show()
        
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(
            os.path.join(
                path, f"swiss_roll_data_{'train' if self.train else 'test'}_{tag}.png"
            )
        )
        plt.clf()

if __name__ == '__main__':
    from matplotlib import animation
    def render_eval_video(origin_line, data_list, video_save_path, video_name, fps=100, dpi=100):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        ims = []
        colors = np.linspace(0, 1, len(data_list))
        
        origin_line_plot = plt.scatter(origin_line[:, 0], origin_line[:, 1], s=1, alpha=0.8, c='yellow', label='Origin Line')

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

    def make_swiss_roll(n_samples=100, noise=0.0, a = 1.5, b = 1):
        generator = np.random.mtrand._rand
        t = a * np.pi * (b + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)

        x = t * np.cos(t)
        z = t * np.sin(t)

        origin_X = np.vstack((x, y, z))
        noisee = noise * generator.standard_normal(size=(3, n_samples))
        X = origin_X + noisee
        X = X.T
        origin_X = origin_X.T
        noisee = noisee.T
        t = np.squeeze(t)

        return X, t, noisee, origin_X

    n_samples = 100
    noise = 0.03
    a = 1.25
    b = 3. / a
    delta_t_barrie = 0.2
    
    X, t, noise_gap, origin_line = make_swiss_roll(
        n_samples=n_samples, noise=noise, a=a, b=b
    )
    t = t.astype(np.float32)
    t = (t - np.min(t)) / (np.max(t) - np.min(t))   # [0, 1]
    x = X.astype(np.float32)[:, [0, 2]]
    origin_line = origin_line.astype(np.float32)[:, [0, 2]]
    noise_gap = noise_gap.astype(np.float32)[:, [0, 2]]
    noise_gap = np.linalg.norm(noise_gap, axis=1)
    
    # transform data
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 10 - 5
    origin_line[:, 0] = origin_line[:, 0] / np.max(np.abs(origin_line[:, 0]))
    origin_line[:, 1] = origin_line[:, 1] / np.max(np.abs(origin_line[:, 1]))
    origin_line = (origin_line - origin_line.min()) / (origin_line.max() - origin_line.min())
    origin_line = origin_line * 10 - 5
    
    # # pair the sampled (x, t) 2by2
    idx_1 = torch.randint(100, (100,))
    idx_2 = torch.randint(n_samples, (100,))
    unfil_x_1 = x[idx_1]
    unfil_t_1 = t[idx_1]
    unfil_x_2 = x[idx_2]
    unfil_t_2 = t[idx_2]
    noise_gap = noise_gap[idx_2]  # next_state
    unfil_delta_t = unfil_t_1 - unfil_t_2
    
    # rule 1: action - delta_t < delta_t_barrie
    idx_fil = np.abs(unfil_delta_t) < delta_t_barrie
    x_1 = unfil_x_1[idx_fil]
    x_2 = unfil_x_2[idx_fil]
    delta_t = unfil_delta_t[idx_fil]
    noise_gap = noise_gap[idx_fil]
    
    # rule 2: distributed stochastic rule
    idx_fil_2 = (np.abs(delta_t) < 0.5 * delta_t_barrie) | (noise_gap < 0.5 * noise)
    x_1 = x_1[idx_fil_2]
    x_2 = x_2[idx_fil_2]
    delta_t = delta_t[idx_fil_2]
    
    # return (x_1, x_2, delta_t, bg, origin_line)
    interpolated_list = np.linspace(x_1, x_2, 1000)
    render_eval_video(origin_line, interpolated_list, "./swiss-roll-nongen-varying-world-model-noise/video", f"eval_video_param_{a}_test", fps=100, dpi=100)