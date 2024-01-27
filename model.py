import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer import *
from layers.positional_encoding import *
from layers.st_lstm import *
from layers.unet import *

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

        # x = F.sigmoid(x)

        # print(x.shape)
        return x
    

class Mjolnir_02(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, kickout=None):
        super(Mjolnir_02, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels = obs_channels
        self.kickout=kickout
        self.num_layers = 2
        mn = 40
        self.mn = mn
        self.num_hidden = [64, 64, 64, 64]
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([64, mn, mn], elementwise_affine=True)
        )
        
        self.encoder_h = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            # nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=1, stride=1).to(torch.device("cuda")),
            #     nn.ReLU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=1, stride=1).to(torch.device("cuda")),
            #     nn.ReLU(),
            # ),
        ])
        self.encoder_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            # nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=1, stride=1).to(torch.device("cuda")),
            #     nn.ReLU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=1, stride=1).to(torch.device("cuda")),
            #     nn.ReLU(),
            # ),
        ])

        # self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([16, mn, mn], elementwise_affine=True)
        )
        # self.decoder_ConvLSTM = ConvLSTM2D(16, 64, kernel_size=5, img_rowcol=mn) # first on is the output channels channels of decoder_1 and second one is the hidden channels

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Sigmoid()
        )

        self.decoder_out = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.cell_list = []
        self.decoder_cell_list = []

        for i in range(self.num_layers):
            self.cell_list.append(
                SpatioTemporalLSTMCell(64, 64, mn, 5, 1, True)
            )
            if i == 0:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(16, 64, mn, 5, 1, True)
              )
            else:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(64, 64, mn, 5, 1, True)
              )

        self.cell_list = nn.ModuleList(self.cell_list)
        self.decoder_cell_list = nn.ModuleList(self.decoder_cell_list)
        self.unet = AttU_Net(1,1)
        self.fusion = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)


    def forward(self, obs):
        batch_size = obs.shape[0]

        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], self.mn, self.mn]).to(torch.device("cuda"))
            h_t.append(zeros)
            c_t.append(zeros)


        memory = torch.zeros([batch_size, self.num_hidden[0], self.mn, self.mn]).to(torch.device("cuda"))



        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[:,t])
            h_t[0], c_t[0], memory = self.cell_list[0](obs_encoder, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            # h, c = self.encoder_ConvLSTM(obs_encoder, h_t[self.num_layers-1], c_t[self.num_layers-1])
        for i in range(self.num_layers):
            h_t[i] = self.encoder_h[i](h_t[i])
            c_t[i] = self.encoder_c[i](c_t[i])
            
        out_list = []
        out_list_radar = []
        last_frame = obs[:, -1, 0:1, :, :]
        
        for t in range(self.future_frames):
            x = self.decoder_1(last_frame)
            h_t[0], c_t[0], memory = self.decoder_cell_list[0](x, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.decoder_cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x =  self.decoder_2(h_t[self.num_layers - 1])
            radar_x = x[:,0:1,:,:] # pick the first channel as radar
            out_list_radar.append(radar_x[:,:,:-1,:-1])
            radar_x = self.unet(radar_x)
            x = self.decoder_out(x)
            x = self.fusion(torch.cat([x, radar_x], dim=1))

            x = x[:,:,:-1,:-1]

            

            out_list.append(x)
            last_frame = F.sigmoid(x)

        # print(pre_frames.shape)
        return torch.cat(out_list, dim=1).unsqueeze(2), torch.cat(out_list_radar, dim=1).unsqueeze(2)



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
        input_batch = input_batch[:,:,0:1,:,:]
        input_batch = input_batch.permute(0, 2, 1, 3, 4)
        result = []
        
        output = self.conv1(input_batch)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        
        output = output.permute(0, 2, 1, 3, 4)
        output = output.reshape(-1, output.shape[2], 159, 159)
        output = self.conv2d_1(output)
        output = self.conv2d_2(output)
        output = output.reshape(-1, 6, output.shape[1], 159, 159)

        return output
        
        for i in range(6):
            x = output[:, :, i, :, :]
            x = self.conv2d_1(x)
            x = self.conv2d_2(x)
            result.append(x)
        
        result = torch.stack(result, dim=2)
        return F.sigmoid(result.permute(0, 2, 1, 3, 4))
        


