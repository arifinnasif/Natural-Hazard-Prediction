import torch
import model

grid_size = 159
input_channels = 8
prev_hours = 6
future_hours = 6
train_model_class = model.Mjolnir_02
test_model_class = model.Mjolnir_02
batch_size = 1
epochs = 1
learning_rate = 0.0001
pos_weight = 20
device = torch.device("cuda")
criterion1_weight = 80
criterion2_weight = 1
criterion3_weight = 1

