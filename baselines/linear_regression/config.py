import torch
# import model

grid_size = 159
prev_hours = 6
future_hours = 6
# train_model_class = model.Mjolnir_08_03
# test_model_class = model.Mjolnir_08_03
batch_size = 64
epochs = 200
learning_rate = 0.0001
pos_weight = 20
device = torch.device("cuda")
criterion1_weight = 1
criterion2_weight = 1
criterion3_weight = 1


list_of_params = [0, 1, 2, 3, 4, 5, 6, 7]
input_channels = len(list_of_params)

def ablation(model, input_batch):
    # input_batch is of shape (batch_size, num_frames, num_channels, height, width)
    # get only the parameters that are in the list from input_batch
    return model(input_batch[:, :, list_of_params, :, :])