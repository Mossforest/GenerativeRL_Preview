import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll


class DynamicSwissRollDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        
        n_samples = config.dataset.n_samples if train else config.dataset.test_n_samples
        pair_samples = config.dataset.pair_samples if train else int(config.dataset.pair_samples * config.dataset.test_n_samples / config.dataset.n_samples)
        delta_t_barrie = config.dataset.delta_t_barrie
        
        
        x_and_t = self.make_swiss_roll(
            n_samples=n_samples, noise=config.dataset.noise, a=config.dataset.a, b=config.dataset.b
        )
        t = x_and_t[1].astype(np.float32)
        t = (t - np.min(t)) / (np.max(t) - np.min(t))   # [0, 1]
        x = x_and_t[0].astype(np.float32)[:, [0, 2]]
        # transform data
        x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
        x[:, 1] = x[:, 1] / np.max(np.abs(x[:, 1]))
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 10 - 5
        
        # pair the sampled (x, t) 2by2
        idx_1 = torch.randint(n_samples, (pair_samples,))
        idx_2 = torch.randint(n_samples, (pair_samples,))
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
        
        self.data_num = x_1.shape[0]
        
        bg = np.repeat(np.array([config.dataset.a, config.dataset.b]), self.data_num, axis=0)
        
        self.data = {
            'state': x_1,
            'next_state': x_2,
            'action': delta_t,
            'background': bg,
            }

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

        X = np.vstack((x, y, z))
        X += noise * generator.standard_normal(size=(3, n_samples))
        X = X.T
        t = np.squeeze(t)

        return X, t

    def plot(self, x, t):
        # plot data with color of value
        plt.scatter(x[:, 0], x[:, 1], c=t, vmin=-5, vmax=3)
        plt.colorbar()
        if not os.path.exists(self.config.parameter.evaluation.video_save_path):
            os.makedirs(self.config.parameter.evaluation.video_save_path)
        plt.savefig(
            os.path.join(
                self.config.parameter.evaluation.video_save_path, f"swiss_roll_data_{'train' if self.train else 'test'}.png"
            )
        )
        plt.clf()
