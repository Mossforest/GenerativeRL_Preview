import os
import random
import numpy as np
import matplotlib.pyplot as plt


def make_swiss_roll(n_samples=100, *, noise=0.0, random_state=None, hole=False, a = 1.5, b = 1):

    t = a * np.pi * (b + 2 * np.random.rand(1, n_samples))
    y = 21 * np.random.rand(1, n_samples)

    x = t * np.cos(t)
    z = t * np.sin(t)

    X = np.vstack((x, y, z))
    # X += noise * generator.standard_normal(size=(3, n_samples))
    X = X.T
    t = np.squeeze(t)

    return X, t




def main(a, b):
    data_num = 1000
    noise = 0.3
    video_save_path=f'./swiss_roll_{a:.2f}_{b:.2f}.png'
    
    
    
    
    x_and_t = make_swiss_roll(
        n_samples=data_num, noise=noise, a=a, b=b
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
    plt.savefig(video_save_path)
    plt.clf()



if __name__ == '__main__':
    # alist = [1.5] + [random.uniform(1, 3) for _ in range(7)]
    # blist = [2] + [random.uniform(0, 5) for _ in range(7)]
    
    alist = np.array([1.5, 1.75, 2, 1.25, 1])
    blist = 3 / alist
    
    print(alist)
    print(blist)
    
    for a, b in zip(alist, blist):
        main(a, b)