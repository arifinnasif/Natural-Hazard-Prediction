import torch
import torch.nn as nn
import config as cfg


# skikit learn linear regression model
from sklearn.linear_model import LinearRegression

class LinearRegressionModel(nn.Module):
    def __init__(self, prediction_horizon, feature_count):
        super(LinearRegressionModel, self).__init__()
        # Defining the input layer
        self.prediction_horizon = prediction_horizon
        self.linear = nn.Linear(feature_count*cfg.grid_size*cfg.grid_size*cfg.prev_hours, cfg.grid_size*cfg.grid_size*prediction_horizon)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x).view(x.size(0), self.prediction_horizon, 1, cfg.grid_size, cfg.grid_size)
