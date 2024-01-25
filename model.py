import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer import *
from layers.positional_encoding import *

class Mjolnir_01(nn.Module):
    def __init__(self):
        super(Mjolnir_01, self).__init__()
        self.num_frames_truth = 8

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ) # output shape: (batch_size, 8, 40, 40)

        self.encoder_conv2d_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ) # output shape: (batch_size, 16, 20, 20)

        self.de_conv2dT_0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 40, 40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)


        self.de_conv_out = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)


        self.positional_encoding = PositionalEncoding3D(channels = 128)


        self.self_attn = AttentionUnit(n_embd=128, n_head=8, block_size=6*20*20, n_layer=6, output_size=128, granularity=20*20, matrix_type="staircase")

        # self.self_attn_temporal = AttentionUnit(n_embd=102400, n_head=8, block_size=6*6, n_layer=6, output_size=102400)

    def vectorize(self, x):
        x = x.permute(0, 1, 3, 4, 2) # (batch_size, seq_len, height, width, channels)
        x = x.reshape(-1, x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]) # (batch_size, seq_len*height*width, channels)
        return x

    def unvectorize(self, x, seq_len, height, width, channels):
        x = x.reshape(-1, seq_len, height, width, channels)
        x = x.permute(0, 1, 4, 2, 3) # (batch_size, seq_len, channels, height, width)
        return x

    def temporal_vectorize(self, x):
        # (batch_size, seq_len, channels, height, width) -> (batch_size, seq_len, channels*height*width)
        return x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4])

    def temporal_unvectorize(self, x, channels, height, width):
        # (batch_size, seq_len, channels*height*width) -> (batch_size, seq_len, channels, height, width)
        return x.reshape(x.shape[0], x.shape[1], channels, height, width)


    def forward(self, input_batch):
        # input_batch is of shape (batch_size, seq_len, channels, height, width)

        # we now need to convert this to (batch_size*seq_len, channels, height, width) to apply the conv2d layers on each frame
        x = input_batch.view(-1, input_batch.shape[2], input_batch.shape[3], input_batch.shape[4])

        x = self.encoder_conv2d_1(x) # (batch_size*seq_len, 16, 80, 80)
        x = self.encoder_conv2d_2(x) # (batch_size*seq_len, 64, 40, 40)
        x = self.encoder_conv2d_3(x) # (batch_size*seq_len, 128, 20, 20)

        # we now get back to the original shape
        x = x.view(input_batch.shape[0], input_batch.shape[1], x.shape[1], x.shape[2], x.shape[3]) # (batch_size, seq_len, 128, 20, 20)

        # apply the positional encoding
        x = x.permute(0, 1, 3, 4, 2) # (batch_size, seq_len, 20, 20, 128)
        x = x + self.positional_encoding(x) # (batch_size, seq_len, 20, 20, 128)
        x = x.permute(0, 1, 4, 2, 3) # (batch_size, seq_len, 128, 20, 20)

        # vectorize the tensor
        x = self.vectorize(x) # (batch_size, seq_len*20*20, 128)

        # apply the self attention layer
        x, _ = self.self_attn(x) # (batch_size, seq_len*20*20, 128)

        # unvectorize the tensor
        x = self.unvectorize(x, 6, 20, 20, 128) # (batch_size, seq_len, 128, 20, 20)

        # apply the decoder conv2d layers
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # (batch_size*seq_len, 128, 20, 20)
        x = self.de_conv2dT_0(x) # (batch_size*seq_len, 64, 40, 40)
        x = self.de_conv2dT_1(x) # (batch_size*seq_len, 32, 80, 80)
        x = self.de_conv2dT_2(x) # (batch_size*seq_len, 32, 160, 160)
        x = self.de_conv_out(x) # (batch_size*seq_len, 1, 160, 160)

        # we now get back to the original shape
        x = x.view(input_batch.shape[0], input_batch.shape[1], x.shape[1], x.shape[2], x.shape[3]) # (batch_size, seq_len, 1, 160, 160)

        x = x[:,:,:,:-1,:-1]

        x = F.sigmoid(x)

        # print(x.shape)
        return x


class StepDeep(nn.Module):
    def __init__(self):
        super(StepDeep, self).__init__()
        self.num_frames_truth = 1
        self.num_frames = 6
        self.fea_dim = 1

        self.conv1 = nn.Sequential( 
            nn.Conv3d(self.num_frames_truth, 128, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(5,1,1), stride=1, padding=(2,0,0)),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            nn.ReLU()
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(7,7), stride=1, padding=(3,3)),
            nn.ReLU()
        )

        self.conv2d_2 = nn.Conv2d(64, 1, kernel_size=(7,7), stride=1, padding=(3,3))
        

    
    def forward(self, input_batch):
        input_batch = input_batch.permute(0, 2, 1, 3, 4)
        result = []
        
        output = self.conv1(input_batch)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.permute(0, 2, 1, 3, 4)
        output = output.view(-1, output.shape[2], 159, 159)
        output = self.conv2d_1(output)
        output = self.conv2d_2(output)
        output = output.view(-1, 6, output.shape[1], 159, 159)

        return F.sigmoid(output)

        output = output.view(-1, 64, 6, 159, 159)
        
        for i in range(6):
            x = output[:, :, i, :, :]
            x = self.conv2d_1(x)
            x = self.conv2d_2(x)
            result.append(x)
        
        result = torch.stack(result, dim=2)
        return result.permute(0, 2, 1, 3, 4)
        


