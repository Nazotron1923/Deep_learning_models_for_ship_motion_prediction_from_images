"""
Different neural network architectures for detecting the line
"""
from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .constants import INPUT_HEIGHT, INPUT_WIDTH
import math

BATCH_SIZE = 32

class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_channel=3, drop_p=0.3, num_output=2):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        self.fc1 = nn.Linear(5376  , 1024) #5376 / 20736
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_output)
        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p= 0.4)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


class ConvolutionalNetwork_p1(nn.Module):
    def __init__(self, num_channel=3, drop_p=0.3, num_output=1024):
        super(ConvolutionalNetwork_p1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )

        self.fc1 = nn.Linear(5376  , num_output) #5376 / 20736
        self.dropout0 = nn.Dropout(p=0.3)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))

        return x

class ConvolutionalNetwork_p2(nn.Module):
    def __init__(self, input_size = 1024, drop_p=0.3, num_output=2):
        super(ConvolutionalNetwork_p2, self).__init__()

        self.fc1 = nn.Linear(input_size, 1024) #5376 / 20736
        # self.fc11 = nn.Linear(5376  , 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_output)
        self.dropout0 = nn.Dropout(p=0.3)

        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p= 0.4)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.tanh(self.fc3(x))

        return x

class CNN_stack_PR_FC (nn.Module):
    def __init__(self, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_PR_FC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.dropout0 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(5376  , 1024) #5376 / 20736

        self.fc2 = nn.Linear(cnn_fc_size, 1024) #5376 / 20736
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)

    def reparameterize(self, mu, logvar, cuda):
        std = logvar.mul(0.5).exp_()
        if cuda:
            esp = torch.randn(*mu.size()).cuda()
        else:
            esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h, cuda):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar, cuda)
        return z

    def encode(self, x, cuda):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h, cuda)
        return z


    def forward(self, x, p_and_roll, num_images, cuda):
        # x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout0(x)
        x = self.encode(x, cuda).view(x.size(0), 1, -1)

        PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]
        PR = torch.cat(PR, 1).view(x.size(0), 1, -1)

        # input_fc = [ th.cat((features[i], PR[i]), 1).view(inputs.size(0), 1, -1) for i in range(num_images)]
        input_fc = torch.cat((x, PR), 2).view(x.size(0), 1, -1)

        x = F.relu(self.fc2(input_fc))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class LSTM_encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=128, num_layers=1):
        super(LSTM_encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, inputs, hiddens):
        outputs, hiddens = self.lstm(inputs, hiddens)
        return outputs, hiddens

    def initHidden(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.hidden_size)


class LSTM_decoder(nn.Module):
    def __init__(self, hidden_size=256, output_size = 20):
        super(LSTM_decoder, self).__init__()
        self.hidden_size  = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out_1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out_2 = nn.Linear(int(hidden_size/2), output_size)


    def forward(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.lstm(outputs, hiddens)
        outputs = F.relu(self.out_1(outputs))
        outputs = torch.tanh(self.out_2(outputs))
        # outputs = self.out_2(outputs)
        return outputs, hiddens

    def initHidden(self, n_batch):
        return torch.zeros(1, n_batch  ,self.hidden_size)


class LSTM_decoder_simple2(nn.Module):
    def __init__(self, hidden_size=256, output_size = 20, num_layers=1 ):
        super(LSTM_decoder_simple2, self).__init__()
        self.hidden_size  = hidden_size
        self.lstm = nn.LSTM(hidden_size, 1000, num_layers ,batch_first=True)
        self.out_1 = nn.Linear(1000, 500)
        self.out_2 = nn.Linear(500, output_size)
        self.dropout0 = nn.Dropout(p=0.2)


    def forward(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.lstm(outputs, hiddens)
        outputs = F.relu(self.out_1(outputs))
        outputs = self.dropout0(outputs)
        outputs = self.out_2(outputs)
        return outputs, hiddens

    def initHidden(self, n_batch):
        return torch.zeros(1, n_batch  ,1000)

class CNN_LSTM_encoder_decoder_images_PR(nn.Module):
    def __init__(self,h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 128, decoder_hidden_size = 256,  output_size = 20):
        super(CNN_LSTM_encoder_decoder_images_PR, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z

    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens

    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)

    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):
        # print("image_s.siwe()- ", image_s.size())
        # print("pr_s.siwe()- ", pr_s.size())
        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        # print("features.siwe()- ", features[0].size())
        # print("PR.siwe()- ", PR[0].size())

        # PR = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class AutoEncoder(nn.Module):
    def __init__(self, num_channel=3, h_dim=2688, z_dim=1024):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding = 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding= 1, output_padding = 0),
            # nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar, cuda):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if cuda:
            esp = esp.cuda()

        z = mu + std * esp
        return z

    def bottleneck(self, h, cuda):
        mu, logvar = self.fc1(h), F.relu(self.fc2(h))
        z = self.reparameterize(mu, logvar, cuda)
        return z

    def encode(self, x, cuda):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h, cuda)
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 32, 7, 12)
        z = self.decoder(z)
        return z

    def forward(self, x, cuda):
        features = self.encode(x, cuda)
        z = self.decode(features)
        return features, z
