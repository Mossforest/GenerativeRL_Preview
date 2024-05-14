import os
from easydict import EasyDict
from rich.progress import track
import numpy as np
import h5py
import time
import random
from typing import Union, Optional


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
import treetensor
from tensordict import TensorDict
from torch.utils.data import Dataset


class MetadriveDataset(Dataset):
    def __init__(self, data_list, key_list):
        self.data_list, self.key_list = self.normalize(data_list, key_list)

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, idx):
        return {key: item[idx] for key, item in zip(self.key_list, self.data_list)}
    
    def normalize(self, data_list, key_list):
        # transverse data to [-1, 1] via activation func
        # action: [-1, 1] naturally
        # state: go tanh
        # dynamics: fixed [min, max] -> [-1, 1]
        #   max_engine_force = [0, 1000]
        #   max_brake_force = [0, 200]
        #   wheel_friction = [0, 1]
        #   max_steering = [0, 90]
        #   max_speed = [0, 100]
        
        # state
        state = data_list[key_list.index('state')]
        data_list[key_list.index('state')] = torch.tanh(state).float()
        next_state = data_list[key_list.index('next_state')]
        data_list[key_list.index('next_state')] = torch.tanh(next_state).float()
        
        # dynamic (* the first elem should be separated into 'dynamic_type')
        dynamic = data_list[key_list.index('dynamic')]
        dyna_max = torch.tensor([1000, 200, 1, 90, 100]).to(dynamic.device)
        dyna_min = torch.tensor([0, 0, 0, 0, 0]).to(dynamic.device)
        
        dynamic_type = dynamic[:, 0]
        key_list.append('dynamic_type')
        data_list.append(dynamic_type)
        
        dyna = dynamic[:, 1:]
        dyna = (dyna_max - dyna) / (dyna_max - dyna_min) * 2 - 1
        data_list[key_list.index('dynamic')] = dyna.float()
        
        # prepare condition = action + dynamic
        action = data_list[key_list.index('action')]
        condition = torch.cat((action, dyna), dim=1).float()
        key_list.append('condition')
        data_list.append(condition)
        
        # prepare blank_condition for CFG & v21
        blank_condition = torch.zeros_like(condition).to(condition.device).float()
        key_list.append('blank_condition')
        data_list.append(blank_condition)
        
        return data_list, key_list


def get_dataset(data_path1, data_path2, train_ratio=0.9):

    data_dict = torch.load(data_path1)
    data_list = list(data_dict.items())
    key_list = [item[0] for item in data_list]
    value_list = [item[1] for item in data_list]
    
    data_dict2 = torch.load(data_path2)
    data_list2 = list(data_dict2.items())
    key_list2 = [item[0] for item in data_list2]
    value_list2 = [item[1] for item in data_list2]

    np.random.seed(0)  # 为了可重现性设置随机种子
    permuted_indices = np.random.permutation(len(value_list[0]))
    train_list = [item[permuted_indices] for item in value_list]
    
    permuted_indices2 = np.random.permutation(len(value_list2[0]))
    eval_list = [item[permuted_indices2] for item in value_list2]
    
    train_dataset = MetadriveDataset(train_list, key_list)
    eval_dataset = MetadriveDataset(eval_list, key_list)
    
    return train_dataset, eval_dataset
