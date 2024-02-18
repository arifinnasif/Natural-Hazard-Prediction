import torch
import model

grid_size = 159
input_channels = 1
prev_hours = 6
future_hours = 6
train_model_class = model.LightNet_O
test_model_class = model.LightNet_O
batch_size = 64
epochs = 100
learning_rate = 0.0001
pos_weight = 20
device = torch.device("cuda")
criterion1_weight = 1
criterion2_weight = 1
criterion3_weight = 1

