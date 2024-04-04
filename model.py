import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer import *
from layers.positional_encoding import *
from layers.st_lstm import *
from layers.unet import *
from layers.conv_lstm import *

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
            nn.Tanh(),
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

        

        # vectorize the tensor
        x = self.vectorize(x) # (batch_size, seq_len*20*20, 128)

        # apply the positional encoding
        # x = x.permute(0, 1, 3, 4, 2) # (batch_size, seq_len, 20, 20, 128)
        x = x + self.positional_encoding(x) # (batch_size, seq_len, 20, 20, 128)
        # x = x.permute(0, 1, 4, 2, 3) # (batch_size, seq_len, 128, 20, 20)

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
    def __init__(self, future_hours, feature_count):
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
        


class LightNet_O(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels):
        super(LightNet_O, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels=obs_channels
        
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ) # (bs, 4, 80, 80)

        self.encoder_ConvLSTM = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=80)

        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ) # (bs, 4, 80, 80)

        self.decoder_ConvLSTM = ConvLSTM2D(4, 64, kernel_size=5, img_rowcol=80) # first on is the output channels channels of decoder_1 and second one is the hidden channels

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            
        )


    def forward(self, obs):
        batch_size = obs.shape[0]

        obs=obs[:,:,0:1]


        h = torch.zeros([batch_size, 8, 80, 80], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, 80, 80], dtype=torch.float32).to(obs.device)


        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[:,t,0:1]) # (bs, 4, 80, 80)
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c) # (bs, 8, 80, 80), (bs, 8, 80, 80)
        h = self.encoder_h(h) # (bs, 64, 80, 80)
        c = self.encoder_c(c) # (bs, 64, 80, 80)


        last_frame = obs[:, -1, 0:1]

        out_list = []

        for t in range(self.future_frames):
            x = self.decoder_1(last_frame) # (bs, 4, 80, 80)
            h, c = self.decoder_ConvLSTM(x, h, c) # (bs, 8, 80, 80), (bs, 8, 80, 80)
            x =  self.decoder_2(c) # (bs, 1, 159, 159)
            out_list.append(x)
            last_frame = F.sigmoid(x)

        return torch.cat(out_list, dim=1).unsqueeze(2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Mjolnir_03(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, kickout=None):
        super(Mjolnir_03, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels = obs_channels
        self.kickout=kickout
        self.num_layers = 2
        mn = 40
        self.mn = mn
        self.num_hidden = [32, 32, 32, 32]
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([32, mn, mn], elementwise_affine=True)
        )
        
        self.encoder_h = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
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
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
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
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Sigmoid()
        )

        self.decoder_out = nn.Conv2d(8, 1, kernel_size=1, stride=1)

        self.cell_list = []
        self.decoder_cell_list = []

        for i in range(self.num_layers):
            self.cell_list.append(
                SpatioTemporalLSTMCell(32, 32, mn, 5, 1, True)
            )
            if i == 0:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(16, 32, mn, 5, 1, True)
              )
            else:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(32, 32, mn, 5, 1, True)
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



class Mjolnir_04(nn.Module):
    """
    Mjolnir_04 is Mjolnir_02 with less parameters
    """
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_04, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels = obs_channels
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
        # self.unet = AttU_Net(1,1)
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
            # radar_x = self.unet(radar_x)
            x = self.decoder_out(x)
            x = self.fusion(torch.cat([x, radar_x], dim=1))

            x = x[:,:,:-1,:-1]

            

            out_list.append(x)
            last_frame = F.sigmoid(x)

        # print(pre_frames.shape)
        return torch.cat(out_list, dim=1).unsqueeze(2), torch.cat(out_list_radar, dim=1).unsqueeze(2)

class ADSNet_O(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels):
        super(ADSNet_O, self).__init__()
        self.num_frames_truth = obs_channels

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ) # output shape: (batch_size, 8, 40, 40)

        self.en_convlstm = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(4, 16, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Encoder
        for t in range(6):

            x = self.encoder_conv2d_1(input_batch[:, t, :, :, :])
            x = self.encoder_conv2d_2(x)
            if t == 0:
                h, c = torch.zeros([batch_size, 8, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 8, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                h, c = self.en_convlstm(x, h, c)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)
   
# model = Mjolnir_04(6, 8).to(torch.device("cuda"))
# a,b = model(torch.rand(1, 6, 8, 159, 159).to(torch.device("cuda")))
# print(a.shape, b.shape)
# print(count_parameters(model))

# model = Mjolnir_02(6, 8).to(torch.device("cuda"))
# print(count_parameters(model))

class Mjolnir_05(nn.Module):
    """adsnet with more layers"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_05, self).__init__()
        self.num_frames_truth = obs_channels

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 32, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(8, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Encoder
        for t in range(6):

            x = self.encoder_conv2d_1(input_batch[:, t, :, :, :])
            x = self.encoder_conv2d_2(x)
            if t == 0:
                h, c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                h, c = self.en_convlstm(x, h, c)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)

class Mjolnir_06(nn.Module):
    """adsnet with more layers"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_06, self).__init__()
        self.num_frames_truth = obs_channels

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ) # output shape: (batch_size, 8, 40, 40)

        self.en_convlstm = ConvLSTM2D(64, 64, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 33, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)

        self.side_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.fusion = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)





        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Encoder
        for t in range(6):

            x = self.encoder_conv2d_1(input_batch[:, t, :, :, :])
            x = self.encoder_conv2d_2(x)
            if t == 0:
                h, c = torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                h, c = self.en_convlstm(x, h, c)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []
        radar_out = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x1, x2 = x[:,0:32,:,:], x[:,32:,:,:]
            x1 = self.de_conv_out(x1)
            x1 = x1[:,:,:-1,:-1]
            x1 = F.tanh(x1)

            x2 = x2[:,:,:-1,:-1]
            radar_out.append(x2)
            x2 = self.side_conv(x2)

            x1 = self.fusion(torch.cat([x1, x2], dim=1))
            out_list.append(x1)
            last_frame = F.sigmoid(x1)


        return torch.cat(out_list, dim=1).unsqueeze(2) , torch.cat(radar_out, dim=1).unsqueeze(2)

class Mjolnir_07(nn.Module):
    """adsnet with more layers"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_07, self).__init__()
        self.num_frames_truth = obs_channels

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([64, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.en_convlstm = ConvLSTM2D(64, 64, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 33, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)

        self.side_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.fusion = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)





        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Encoder
        for t in range(6):

            x = self.encoder_conv2d_1(input_batch[:, t, :, :, :])
            x = self.encoder_conv2d_2(x)
            if t == 0:
                h, c = torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                h, c = self.en_convlstm(x, h, c)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []
        radar_out = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x1, x2 = x[:,0:32,:,:], x[:,32:,:,:]
            x1 = self.de_conv_out(x1)
            x1 = x1[:,:,:-1,:-1]
            x1 = F.tanh(x1)

            x2 = x2[:,:,:-1,:-1]
            radar_out.append(x2)
            x2 = self.side_conv(x2)

            x1 = self.fusion(torch.cat([x1, x2], dim=1))
            out_list.append(x1)
            last_frame = F.sigmoid(x1)


        return torch.cat(out_list, dim=1).unsqueeze(2) , torch.cat(radar_out, dim=1).unsqueeze(2)

class Mjolnir_08(nn.Module):
    """dual encoder"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_08, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.lightning_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Other Encoder
        self.other_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth-3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.other_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.other_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        h = torch.cat([lightning_h, other_h], dim=1)
        c = torch.cat([lightning_c, other_c], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)

class Mjolnir_09(nn.Module):
    """dual encoder with spatio encoder"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_09, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.lightning_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Other Encoder
        self.other_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth-3-1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.other_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.other_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Spatio Encoder
        self.spatio_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.spatio_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        

        self.en_de_h = nn.Sequential(
            nn.Conv2d(68, 68, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(68, 68, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 68, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(68, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:7, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        # Spatio Encoder
        x = self.spatio_encoder_conv2d_1(input_batch[:, -1, 7:, :, :])
        x = self.spatio_encoder_conv2d_2(x)

        h = torch.cat([lightning_h, other_h, x[:,:4,:,:]], dim=1)
        c = torch.cat([lightning_c, other_c, x[:,4:,:,:]], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)

class Mjolnir_10(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_10, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        mn = (159//2)//2
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels, 4*obs_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4*obs_channels, 4*obs_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4*obs_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([64, mn, mn], elementwise_affine=True)
        )
        self.encoder_ConvLSTM = ConvLSTM2D(64, 64, kernel_size=5, img_rowcol=mn)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        # self.decoder_ConvLSTM = ConvLSTM2D(8, 8, kernel_size=5, img_rowcol=mn)

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([16, mn, mn], elementwise_affine=True)
        )
        self.decoder_ConvLSTM = ConvLSTM2D(16, 64, kernel_size=5, img_rowcol=mn) # first on is the output channels channels of decoder_1 and second one is the hidden channels

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # 7 -> 5
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            # nn.ReLU(),
            # nn.Sigmoid()
        )

        self.sig = nn.Sigmoid()

        # self.decoder_3 = nn.Sequential(
        #     nn.Conv2d(64, obs_channels, kernel_size=1, stride=1),
        #     nn.ReLU()
        # )

        # self.decoder_3 = nn.Sequential(
        #     nn.Conv2d(obs_channels, 1, kernel_size=1, stride=1),
        #     nn.ReLU()
        # )
        self.conv_fusion_h = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, groups=2)





    def forward(self, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        # obs = obs.permute(1, 0, 2, 3, 4).contiguous()

        # [batch_size, frames, channels, x, y]


        batch_size = obs.shape[0]


        h = torch.zeros([batch_size, 64, (159//2)//2, (159//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 64, (159//2)//2, (159//2)//2], dtype=torch.float32).to(obs.device)

        pre_frames = torch.zeros([batch_size, self.future_frames, 1, 159, 159], dtype=torch.float32).to(obs.device)


        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[:,t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)

        # print(h.shape)
        # print(c.shape)


        x = obs[:, -1, 0:1]
        # print("last frame shape", x.shape)
        for t in range(self.future_frames):
            x = self.decoder_1(x)
            # print("decoder_1 shape", x.shape)
            h, c = self.decoder_ConvLSTM(x, h, c)
            # print("decoder_ConvLSTM shape", h.shape, c.shape)
            x =  self.decoder_2(c)
            # x=self.sig(x)
            # print("decoder_2 shape", x.shape)
            pre = x
            # print("decoder_3 shape", x.shape)
            # pre = self.decoder_4(x)
            # print("pre shape", pre.shape)
            pre_frames[:,t] = pre
            # x = pre

        # print(pre_frames.shape)
        return pre_frames

class Mjolnir_08_02(nn.Module):
    """dual encoder with normalized inputs"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_08_02, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.lightning_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Other Encoder
        self.other_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth-3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.other_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        ) # output shape: (batch_size, 8, 40, 40)

        self.other_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]
        input_batch = normalize_input(input_batch)

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        h = torch.cat([lightning_h, other_h], dim=1)
        c = torch.cat([lightning_c, other_c], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)

class Mjolnir_08_03(nn.Module):
    """dual encoder with dropout"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_08_03, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            nn.Dropout(0.6)
        ) # output shape: (batch_size, 4, 80, 80)

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(0.6),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True),
        ) # output shape: (batch_size, 8, 40, 40)

        self.lightning_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Other Encoder
        self.other_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth-3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            nn.Dropout(0.6)
        ) # output shape: (batch_size, 4, 80, 80)

        self.other_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(0.6),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True),
        ) # output shape: (batch_size, 8, 40, 40)

        self.other_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            nn.Dropout(0.6)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(0.6),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.6)
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.6)
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)


        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        h = torch.cat([lightning_h, other_h], dim=1)
        c = torch.cat([lightning_c, other_c], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)

class Mjolnir_08_04(nn.Module):
    """dual encoder with more layers in fusion"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_08_04, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            # nn.Dropout(0.2)
        ) # output shape: (batch_size, 4, 80, 80)

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # nn.Dropout(0.2),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True),
        ) # output shape: (batch_size, 8, 40, 40)

        self.lightning_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        # Other Encoder
        self.other_encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth-3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            # nn.Dropout(0.2)
        ) # output shape: (batch_size, 4, 80, 80)

        self.other_encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # nn.Dropout(0.2),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True),
        ) # output shape: (batch_size, 8, 40, 40)

        self.other_en_convlstm = ConvLSTM2D(32, 32, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0),
            nn.ReLU(),
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
            # nn.Dropout(0.2)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # nn.Dropout(0.2),
            nn.LayerNorm([8, 40, 40], elementwise_affine=True)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(8, 64, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            # nn.Dropout(0.2)
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(),
            # nn.Dropout(0.2)
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)


        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 32, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        h = torch.cat([lightning_h, other_h], dim=1)
        c = torch.cat([lightning_c, other_c], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)


class Mjolnir_11(nn.Module):
    """dual encoder with st lstm"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(Mjolnir_11, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels = obs_channels
        self.num_layers = 2
        mn = 40
        self.mn = mn
        self.num_hidden = [32, 32, 32, 32]
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([32, mn, mn], elementwise_affine=True)
        )

        self.other_encoder_module = nn.Sequential(
            nn.Conv2d(obs_channels-3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([32, mn, mn], elementwise_affine=True)
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
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([8, mn, mn], elementwise_affine=True)
        )
        # self.decoder_ConvLSTM = ConvLSTM2D(16, 64, kernel_size=5, img_rowcol=mn) # first on is the output channels channels of decoder_1 and second one is the hidden channels

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Sigmoid()
        )

        self.decoder_out = nn.Conv2d(16, 1, kernel_size=1, stride=1)

        self.obs_cell_list = []
        self.other_cell_list = []
        self.decoder_cell_list = []

        for i in range(self.num_layers):
            self.obs_cell_list.append(
                SpatioTemporalLSTMCell(32, 32, mn, 5, 1, True)
            )

            self.other_cell_list.append(
                SpatioTemporalLSTMCell(32, 32, mn, 5, 1, True)
            )

            if i == 0:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(8, 64, mn, 5, 1, True)
              )
            else:
              self.decoder_cell_list.append(
                  SpatioTemporalLSTMCell(64, 64, mn, 5, 1, True)
              )

        self.obs_cell_list = nn.ModuleList(self.obs_cell_list)
        self.other_cell_list = nn.ModuleList(self.other_cell_list)
        self.decoder_cell_list = nn.ModuleList(self.decoder_cell_list)
        # self.unet = AttU_Net(1,1)
        # self.fusion = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2)


    def forward(self, obs):
        batch_size = obs.shape[0]

        obs_h_t = []
        obs_c_t = []

        other_h_t = []
        other_c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch_size, self.num_hidden[i], self.mn, self.mn]).to(torch.device("cuda"))
            obs_h_t.append(zeros)
            obs_c_t.append(zeros)

            other_h_t.append(zeros)
            other_c_t.append(zeros)


        obs_memory = torch.zeros([batch_size, self.num_hidden[0], self.mn, self.mn]).to(torch.device("cuda"))
        other_memory = torch.zeros([batch_size, self.num_hidden[0], self.mn, self.mn]).to(torch.device("cuda"))



        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[:,t,0:3,:,:])
            other_encoder = self.other_encoder_module(obs[:,t,3:,:,:])
            obs_h_t[0], obs_c_t[0], obs_memory = self.obs_cell_list[0](obs_encoder, obs_h_t[0], obs_c_t[0], obs_memory)
            other_h_t[0], other_c_t[0], other_memory = self.other_cell_list[0](other_encoder, other_h_t[0], other_c_t[0], other_memory)

            for i in range(1, self.num_layers):
                obs_h_t[i], obs_c_t[i], obs_memory = self.obs_cell_list[i](obs_h_t[i - 1], obs_h_t[i], obs_c_t[i], obs_memory)
                other_h_t[i], other_c_t[i], other_memory = self.other_cell_list[i](other_h_t[i - 1], other_h_t[i], other_c_t[i], other_memory)

            # h, c = self.encoder_ConvLSTM(obs_encoder, h_t[self.num_layers-1], c_t[self.num_layers-1])
        h_t = []
        c_t = []
            
        for i in range(self.num_layers):
            h_t.append(self.encoder_h[i](torch.cat([obs_h_t[i], other_h_t[i]], dim=1)))
            c_t.append(self.encoder_c[i](torch.cat([obs_c_t[i], other_c_t[i]], dim=1)))
            
        out_list = []
        # out_list_radar = []
        last_frame = obs[:, -1, 0:1, :, :]
        memory = torch.concat([obs_memory, other_memory], dim=1)
        
        for t in range(self.future_frames):
            x = self.decoder_1(last_frame)
            h_t[0], c_t[0], memory = self.decoder_cell_list[0](x, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.decoder_cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x =  self.decoder_2(h_t[self.num_layers - 1])
            # radar_x = x[:,0:1,:,:] # pick the first channel as radar
            # out_list_radar.append(radar_x[:,:,:-1,:-1])
            # radar_x = self.unet(radar_x)
            x = self.decoder_out(x)
            # x = self.fusion(torch.cat([x, radar_x], dim=1))

            x = x[:,:,:-1,:-1]

            

            out_list.append(x)
            last_frame = F.sigmoid(x)

        # print(pre_frames.shape)
        return torch.cat(out_list, dim=1).unsqueeze(2)#, torch.cat(out_list_radar, dim=1).unsqueeze(2)

class NewLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.BCELoss = nn.BCELoss(reduction='none')

  def forward(self, prediction, truth, transformed_ground_truth):
    # as prediction is not sigmoided
    prediction = torch.sigmoid(prediction)
    weight = (1-transformed_ground_truth)*prediction + transformed_ground_truth*(1-prediction)
    loss = self.BCELoss(prediction, truth)
    loss = loss * weight
    return torch.mean(loss)
  
def normalize_input(input_tensor, axes=(1, 3, 4)):
    # Normalize over time steps, height, and width for features index 1 to 7
    batch_size, time_steps, features, height, width = input_tensor.shape

    

    # Step 1: Select the subset of features (from 1 to 7)
    subset_tensor = input_tensor[:, :, 1:8, :, :]

    

    # Step 2: Calculate the mean and standard deviation
    mean = subset_tensor.mean(dim=(1, 3, 4), keepdim=True)  # Mean over time steps, height, and width
    std = subset_tensor.std(dim=(1, 3, 4), keepdim=True)    # Std over time steps, height, and width


    # Step 3: Normalize
    normalized_tensor = (subset_tensor - mean) / (std + 1e-6)  # Adding a small value to avoid division by zero

    # Replace the original features with the normalized ones
    input_tensor[:, :, 1:8, :, :] = normalized_tensor
    return input_tensor


# print(count_parameters(Mjolnir_11(6, 8)))
class LinearRegressionModel(nn.Module):
    def __init__(self, prediction_horizon, feature_count):
        super(LinearRegressionModel, self).__init__()
        # Defining the input layer
        self.lr = nn.Linear(6*feature_count, prediction_horizon)
        self.prediction_horizon = prediction_horizon
    
    def forward(self, x):
        # permute to batch_size, height, width, channels, time_steps
        x = x.permute(0, 3, 4, 2, 1)
        # Flatten the input tensor
        x = x.reshape(-1, x.shape[3]* x.shape[4])
        # Pass the input tensor through the linear layer
        out = self.lr(x)
        out = out.reshape(-1, 159, 159, 1, self.prediction_horizon)
        # permute to batch_size, time_steps, channels, height, width
        out = out.permute(0, 4, 3, 1, 2)
        return out