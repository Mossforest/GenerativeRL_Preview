import gymnasium as gym
# import gym as gym
import multiprocessing as mp
import os
import cv2

import torch

from easydict import EasyDict
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from grl.utils import set_seed
import h5py

image_size = 256
config = EasyDict(
    dict(
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        data=dict(
            image_size=image_size,
            image_path="/mnt/d/Dataset/atari_world_model_1/images/",
            data_path="/mnt/d/Dataset/atari_world_model_1/",
            origin_image_size=(210, 160, 3),
        ),
    )
)

def resize_image(image, size):
    # image is numpy array of shape (210, 160, 3), resize to (image_size, image_size, 3)
    return cv2.resize(image[:, :, ::-1], (size, size), interpolation=cv2.INTER_AREA)

def save_image(image, path):
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def ramdon_action(env):
    action = env.action_space.sample()
    return action


def collect_data(env_id, i, data_path, image_path):
    try:
        obs_list = []
        reward_list = []
        action_list = []
        next_obs_list = []
        done_list = []

        env = gym.make(env_id, render_mode="rgb_array")
        obs = env.reset()

        counter = 0
        save_image(
            resize_image(env.render(), config.data.image_size),
            os.path.join(image_path, f"{i}_{counter}.png"),
        )

        terminated, truncated = False, False
        
        while not terminated and not truncated:
            action = ramdon_action(env)

            if isinstance(obs, tuple):
                obs = obs[0]
            else:
                obs = obs
            assert obs.shape == (config.data.origin_image_size[0], config.data.origin_image_size[1], config.data.origin_image_size[2]), "obs shape is not correct"
            obs_list.append(torch.tensor(obs, dtype=torch.int8).to(config.device).unsqueeze(0))

            obs, reward, terminated, truncated, info = env.step(action)
            
            reward_list.append(torch.tensor(reward, dtype=torch.float32).to(config.device))
            action_list.append(torch.tensor(action, dtype=torch.int8).to(config.device))
            assert obs.shape == (config.data.origin_image_size[0], config.data.origin_image_size[1], config.data.origin_image_size[2]), "obs shape is not correct"
            next_obs_list.append(torch.tensor(obs, dtype=torch.int8).to(config.device).unsqueeze(0))
            done_list.append(torch.tensor(terminated or truncated, dtype=torch.int8).to(config.device))

            counter += 1
            save_image(
                resize_image(env.render(), config.data.image_size),
                os.path.join(image_path, f"{i}_{counter}.png"),
            )

        env.close()

        obs_tensor = torch.concatenate(obs_list)
        reward_tensor = torch.tensor(reward_list)
        action_tensor = torch.tensor(action_list)
        next_obs_tensor = torch.concatenate(next_obs_list)
        done_tensor = torch.tensor(done_list)

        return obs_tensor, reward_tensor, action_tensor, next_obs_tensor, done_tensor

    except Exception as e:
        print(e)
        return None, None, None, None, None



class H5PYDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(data_path)
        self.condition_length = 1

        # open all files and record the episode length by key "episode_length"
        self.episode_length = []
        # legal_episode_length is the episode length that is larger than condition_length
        self.legal_episode_length = []
        for file_name in self.file_list:
            if not file_name.endswith(".h5"):
                continue
            with h5py.File(os.path.join(data_path, file_name), "r") as f:
                self.episode_length.append(f["episode_length"][()])
                if f["episode_length"][()] > self.condition_length:
                    self.legal_episode_length.append(f["episode_length"][()]-self.condition_length+1)
                else:
                    print(f"episode length is {f['episode_length'][()]}, which is less than condition length {self.condition_length}")
                    self.legal_episode_length.append(0)
        
        # calculate the sum of legal episode length
        self.legal_episode_length_sum = []
        self.legal_episode_length_sum.append(0)
        for i in range(1, len(self.legal_episode_length)):
            self.legal_episode_length_sum.append(sum(self.legal_episode_length[:i]))

        self.total_length = sum(self.legal_episode_length)
        self.index = torch.zeros((self.total_length, 2), dtype=torch.int64)
        for i in range(len(self.legal_episode_length)):
            if self.legal_episode_length[i] > 0:
                self.index[self.legal_episode_length_sum[i]:self.legal_episode_length_sum[i] + self.legal_episode_length[i], 0] = i
                self.index[self.legal_episode_length_sum[i]:self.legal_episode_length_sum[i] + self.legal_episode_length[i], 1] = torch.arange(self.legal_episode_length[i])



    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_order, episode_order = self.index[idx]
        with h5py.File(os.path.join(self.data_path, self.file_list[file_order]), "r") as f:
            obs = torch.tensor(f["obs"][episode_order:episode_order + self.condition_length])
            reward = torch.tensor(f["reward"][episode_order:episode_order + self.condition_length])
            action = torch.tensor(f["action"][episode_order:episode_order + self.condition_length])
            next_obs = torch.tensor(f["next_obs"][episode_order:episode_order + self.condition_length])
            done = torch.tensor(f["done"][episode_order:episode_order + self.condition_length])

        return obs, reward, action, next_obs, done


if __name__ == "__main__":
    set_seed()
    mp.set_start_method("spawn")

    if not os.path.exists(config.data.data_path):
        os.makedirs(config.data.data_path)

    collect = True

    if collect:

        for env_id in [
            "ALE/MsPacman-v5"
        ]:

            i = 0
            while i < 10:
                obs_tensor, reward_tensor, action_tensor, next_obs_tensor, done_tensor = collect_data(env_id, i, config.data.data_path, config.data.image_path)
                if obs_tensor is not None:
                    with h5py.File(os.path.join(config.data.data_path, f"{i}.h5"), "w") as f:
                        f.create_dataset("obs", data=obs_tensor.cpu().numpy())
                        f.create_dataset("reward", data=reward_tensor.cpu().numpy())
                        f.create_dataset("action", data=action_tensor.cpu().numpy())
                        f.create_dataset("next_obs", data=next_obs_tensor.cpu().numpy())
                        f.create_dataset("done", data=done_tensor.cpu().numpy())
                        f.create_dataset("episode_length", data=len(obs_tensor))
                    i += 1


    dataset = H5PYDataset(config.data.data_path)
    print(len(dataset))
    print(dataset[0])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for obs, reward, action, next_obs, done in data_loader:
        print(obs.shape, reward.shape, action.shape, next_obs.shape, done.shape)
        break
