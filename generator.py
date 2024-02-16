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
import config as cfg

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
    
class TrainDataset(Dataset):
    # all of 2021 and 2022
    def __init__(self):
        self.data_2021 = np.load("dataset/total_data_2021.npy")
        self.data_2022 = np.load("dataset/total_data_2022.npy")
        self.total_hours = self.data_2021.shape[0] + self.data_2022.shape[0]
        self.feature_count = self.data_2021.shape[1]
        self.dim = self.data_2021.shape[2]
        self.past_hours = 6
        self.future_hours = 6
        
        # self.data = np.random.rand(self.total_hours, self.feature_count, self.dim, self.dim)

    def __len__(self):
        return (self.data_2021.shape[0] - self.past_hours - self.future_hours + 1) + (self.data_2022.shape[0] - self.past_hours - self.future_hours + 1)

    def __getitem__(self, idx):
        idx_old = idx
        if idx < 720+744+720+744 - self.past_hours - self.future_hours + 1:
            X = self.data_2021[(idx):(idx+self.past_hours),:,:,:]
            y = self.data_2021[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
            y_aux = self.data_2021[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        else:
            idx = idx - (720+744+720+744 - self.past_hours - self.future_hours + 1)
            X = self.data_2022[(idx):(idx+self.past_hours),:,:,:]
            y = self.data_2022[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
            y_aux = self.data_2022[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx_old


class ValidationDataset(Dataset):
    # first two months of 2023
    def __init__(self):
        self.data_2023 = np.load("dataset/total_data_2023.npy")
        self.data_2023 = self.data_2023[:720+744,:,:,:]
        self.total_hours = self.data_2023.shape[0]
        self.feature_count = self.data_2023.shape[1]
        self.dim = self.data_2023.shape[2]
        self.past_hours = 6
        self.future_hours = 6
        
        # self.data = np.random.rand(self.total_hours, self.feature_count, self.dim, self.dim)

    def __len__(self):
        return self.total_hours - self.past_hours - self.future_hours + 1

    def __getitem__(self, idx):
        X = self.data_2023[(idx):(idx+self.past_hours),:,:,:]
        y = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
        y_aux = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx
    

class TestDataset(Dataset):
    # last two month of 2023
    def __init__(self):
        self.data_2023 = np.load("dataset/total_data_2023.npy")
        self.data_2023 = self.data_2023[720+744:,:,:,:]
        self.total_hours = self.data_2023.shape[0]
        self.feature_count = self.data_2023.shape[1]
        self.dim = self.data_2023.shape[2]
        self.past_hours = 6
        self.future_hours = 6
        
        # self.data = np.random.rand(self.total_hours, self.feature_count, self.dim, self.dim)

    def __len__(self):
        return self.total_hours - self.past_hours - self.future_hours + 1

    def __getitem__(self, idx):
        X = self.data_2023[(idx):(idx+self.past_hours),:,:,:]
        y = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
        y_aux = self.data_2023[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx


def get_train_loader():
    train_dataset = TrainDataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    return train_loader

def get_val_loader():
    val_dataset = ValidationDataset()
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return val_loader

def get_test_loader():
    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return test_loader
