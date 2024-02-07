import config as cfg
import torch
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from score import *

model_path = sys.argv[1]
dataset_path = sys.argv[2]

class TestDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path)
        self.total_hours = self.data.shape[0]
        self.feature_count = self.data.shape[1]
        self.past_hours = 6
        self.future_hours = 6

    def __len__(self):
        return self.total_hours - self.past_hours - self.future_hours + 1

    def __getitem__(self, idx):
        X = self.data[(idx):(idx+self.past_hours),:,:,:]
        y = self.data[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
        y_aux = self.data[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx


test_dataset = TestDataset(dataset_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)



model  = cfg.test_model_class(cfg.prev_hours, cfg.input_channels)
model.load_state_dict(torch.load(model_path))
model.float()
model.to(cfg.device)

model.eval()

model_eval_testdata = Model_eval(is_save_model=False)


val_calparams_epoch = Cal_params_epoch()
for i, (X, y, y_aux, idx) in enumerate(test_loader):
    X = X.float().to(cfg.device)
    y = y.float().to(cfg.device)
    # print("hi"+str(i))
    predicted_frames, _ = model(X)
    # print(predicted_frames.shape)

    # output
    pod, far, ts, ets = val_calparams_epoch.cal_batch(y, predicted_frames)
    sumpod, sumfar, sumts, sumets = val_calparams_epoch.cal_batch_sum(y, predicted_frames)
    del X
    del y
    del _
    del predicted_frames
sumpod, sumfar, sumts, sumets = val_calparams_epoch.cal_epoch_sum()
pod, far, ts, ets = val_calparams_epoch.cal_epoch()


# spatio-temporal metrics
print("Spatio-temporal metrics")
print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1, val_calparams_epoch.n2, val_calparams_epoch.n3, val_calparams_epoch.n4))
print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod, far, ts, ets))

# spatio-only metrics
print("Spatio-only metrics")
print('TP:{:09d}  FP:{:09d} \FN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1sum, val_calparams_epoch.n2sum, val_calparams_epoch.n3sum, val_calparams_epoch.n4sum))
print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod, sumfar, sumts, sumets))


