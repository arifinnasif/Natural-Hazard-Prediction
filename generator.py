import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self):
        self.total_hours = 720+744+720+744+720+744+720+744
        self.feature_count = 8
        self.dim = 159
        self.past_hours = 6
        self.future_hours = 6
        self.data_2022 = np.load("dataset/total_data_2022.npy")
        self.data_2023 = np.load("dataset/total_data_2023.npy")
        # self.data = np.random.rand(self.total_hours, self.feature_count, self.dim, self.dim)

    def __len__(self):
        return 2*(720+744+720+744 - self.past_hours - self.future_hours)

    def __getitem__(self, idx):
        idx_old = idx
        if idx < 720+744+720+744 - self.past_hours - self.future_hours:
            X = self.data_2022[(idx):(idx+self.past_hours),:,:,:]
            y = self.data_2022[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
            y_aux = self.data_2022[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        else:
            idx = idx - (720+744+720+744 - self.past_hours - self.future_hours)
            X = self.data_2023[(idx):(idx+self.past_hours),:,:,:]
            y = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
            y_aux = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx_old


# full_dataset = CustomDataset()

# train_size = int(0.7 * len(full_dataset))
# val_size = int(0.2 * len(full_dataset))
# test_size = len(full_dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
