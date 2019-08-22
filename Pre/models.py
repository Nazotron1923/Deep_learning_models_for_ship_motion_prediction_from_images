"""

Architectures of different neural models to solve the problem

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, cuda = True, num_channel=3, h_dim=2688, z_dim=1024):
        super(AutoEncoder, self).__init__()
        self.cuda_p = cuda
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
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )
        # self.fc0 = nn.Linear(h_dim, int(h_dim/2))
        # self.dropout0 = nn.Dropout(p=0.1)
        # self.fc00 = nn.Linear(int(h_dim/2), int(h_dim/2))
        # self.dropout00 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)


        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
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


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.relu(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 32, 7, 12)
        z = self.decoder(z)
        return z


    def forward(self, x):
        features = self.encode(x)
        z = self.decode(features)
        return features, z


class CNN_stack_FC_first(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20):
        super(CNN_stack_FC_first, self).__init__()
        self.cuda_p = cuda
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
        self.fc1 = nn.Linear(5376  , cnn_fc_size) #5376 / 20736
        self.fc2 = nn.Linear(cnn_fc_size, 128)
        self.fc3 = nn.Linear(128, num_output)
        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p= 0.4)


    def forward(self, x, p_and_roll, num_images):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x =  torch.tanh(self.fc3(x)).view(x.size(0), -1, 2)

        return x


class CNN_stack_FC(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_FC, self).__init__()
        self.cuda_p = cuda
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
        self.fc1 = nn.Linear(5376  , cnn_fc_size) #5376 / 20736

        self.fc2 = nn.Linear(cnn_fc_size, 512) #5376 / 20736
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
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


    def forward(self, x, p_and_roll, num_images):

        x = self.encode(x).view(x.size(0), 1, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class CNN_stack_PR_FC(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_PR_FC, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),  # 8x54x96
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),              # 8x27x48
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),           # 16x27x48
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),                 # 16x14x24
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32x14x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),                 # 32x7x12
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


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
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


    def forward(self, x, p_and_roll, num_images):

        x = self.encode(x).view(x.size(0), 1, -1)

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


class CNN_PR_FC (nn.Module):
    def __init__(self, cuda = True, cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_PR_FC, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
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

        self.fc2 = nn.Linear(cnn_fc_size, int(cnn_fc_size/2)) #5376 / 20736
        self.fc22 = nn.Linear(int(cnn_fc_size/2), 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, num_output)
        self.dropout1 = nn.Dropout(p=0.3)

        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p= 0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
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


    def forward(self, x, p_and_roll, num_images):

        features = [self.encode(x[:,i,:,:,:]) for i in range(num_images-1, -1, -1)]

        PR = [p_and_roll[:,i,:] for i in range(num_images-1, -1, -1)]

        input_fc = [ torch.cat((features[i], PR[i]), 1).view(x.size(0), 1, -1) for i in range(num_images)]
        input_fc = torch.cat(input_fc, 2).view(x.size(0), 1, -1)

        x = F.relu(self.fc2(input_fc))
        x = self.dropout2(x)
        x = F.relu(self.fc22(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x


class LSTM_encoder_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


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

        PR  = [pr_s[i] for i in range(use_n_im)]
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


# class CNN_LSTM_encoder_decoder_images_PR (nn.Module):
#     def __init__(self,h_dim=2688, z_dim=400, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
#         super(CNN_LSTM_encoder_decoder_images_PR, self).__init__()
#         self.encoder_hidden_size = encoder_hidden_size
#         self.decoder_hidden_size = decoder_hidden_size
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
#             nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
#         )
#
#         self.fc0 = nn.Linear(h_dim, int(h_dim/2))
#         self.dropout0 = nn.Dropout(p=0.1)
#         self.fc00 = nn.Linear(int(h_dim/2), int(h_dim/2))
#         self.dropout00 = nn.Dropout(p=0.05)
#
#         self.mu = nn.Linear(int(h_dim/2), z_dim)
#         self.std = nn.Linear(int(h_dim/2), z_dim)
#
#         self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
#
#         self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)
#
#         self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
#         self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)
#
#
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#
#         esp = torch.randn(*mu.size()).cuda()
#         z = mu + std * esp
#         return z
#
#
#     def bottleneck(self, h):
#         mu  = self.mu(h)
#         logvar = F.relu(self.std(h))
#         z = self.reparameterize(mu, logvar)
#         return z
#
#
#     def encode(self, x):
#         h = self.encoder(x)
#         h = h.view(h.size(0), -1)
#         h = self.dropout0(F.relu(self.fc0(h)))
#         h = self.dropout00(F.relu(self.fc00(h)))
#         z = self.bottleneck(h)
#         return z
#
#
#     def LSTM_encoder(self, inputs, hiddens):
#         outputs, hiddens = self.encoder_lstm(inputs, hiddens)
#         return outputs, hiddens
#
#
#     def LSTM_decoder(self, inputs, hiddens):
#         outputs = F.relu(inputs)
#         outputs, hiddens = self.decoder_lstm(outputs, hiddens)
#         outputs = F.relu(self.decoder_fc_1(outputs))
#         outputs = torch.tanh(self.decoder_fc_2(outputs))
#         return outputs, hiddens
#
#
#     def initHiddenEncoder(self, n_batch):
#         return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)
#
#
#     def initHiddenDecoder(self, n_batch):
#         return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)
#
#
#     def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):
#
#         features = [self.encode( image_s[i] ) for i in range(use_n_im)]
#         PR  = [pr_s[i] for i in range(use_n_im)]
#
#         lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
#         lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)
#
#         encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
#         decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)
#
#         decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)
#
#         return decoder_output, encoder_hidden, decoder_hidden
#
#
# class AutoEncoder(nn.Module):
#     def __init__(self, num_channel=3, h_dim=2688, z_dim=400):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),
#             nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
#             # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             # nn.BatchNorm2d(64),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
#         )
#         self.fc0 = nn.Linear(h_dim, int(h_dim/2))
#         self.dropout0 = nn.Dropout(p=0.1)
#         self.fc00 = nn.Linear(int(h_dim/2), int(h_dim/2))
#         self.dropout00 = nn.Dropout(p=0.05)
#         self.fc1 = nn.Linear(int(h_dim/2), z_dim)
#         self.fc2 = nn.Linear(int(h_dim/2), z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)
#
#
#         self.decoder = nn.Sequential(
#             # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
#             # nn.BatchNorm2d(32),
#             # nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding = 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding= 1, output_padding = 0),
#             nn.BatchNorm2d(3),
#             nn.Tanh()
#         )
#
#
#     def reparameterize(self, mu, logvar, cuda):
#         std = logvar.mul(0.5).exp_()
#         esp = torch.randn(*mu.size())
#         if cuda:
#             esp = esp.cuda()
#
#         z = mu + std * esp
#         return z
#
#
#     def bottleneck(self, h, cuda):
#         mu, logvar = self.fc1(h), F.relu(self.fc2(h))
#         z = self.reparameterize(mu, logvar, cuda)
#         return z
#
#
#     def encode(self, x, cuda):
#         h = self.encoder(x)
#         h = h.view(h.size(0), -1)
#         h = self.dropout0(F.relu(self.fc0(h)))
#         h = self.dropout00(F.relu(self.fc00(h)))
#         z = self.bottleneck(h, cuda)
#         return z
#
#
#     def decode(self, z):
#         z = self.fc3(z)
#         z = z.view(z.size(0), 32, 7, 12)
#         z = self.decoder(z)
#         return z
#
#
#     def forward(self, x, cuda):
#         features = self.encode(x, cuda)
#         z = self.decode(features)
#         # print("/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////size -> ", z.size())
#         return features, z


class CNN_LSTM_encoder_decoder_images_PR (nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 300,  decoder_input_size = 300, decoder_hidden_size = 150, output_size = 20):
        super(CNN_LSTM_encoder_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
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

        self.mu = nn.Linear(int(h_dim), z_dim)
        self.std = nn.Linear(int(h_dim), z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
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

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_encoder_decoder_images(nn.Module):
    def __init__(self,cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 20):
        super(CNN_LSTM_encoder_decoder_images, self).__init__()
        self.cuda_p = cuda
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

        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
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

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden


class CNN_LSTM_image_encoder_PR_encoder_decoder(nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, im_encoder_input_size = 4096, pr_encoder_input_size = 20 , im_encoder_hidden_size = 128, pr_encoder_hidden_size = 128, decoder_hidden_size = 256,  output_size = 20):
        super(CNN_LSTM_image_encoder_PR_encoder_decoder, self).__init__()
        self.cuda_p = cuda
        self.im_encoder_hidden_size = im_encoder_hidden_size
        self.pr_encoder_hidden_size = pr_encoder_hidden_size
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

        self.im_encoder_lstm = nn.LSTM(im_encoder_input_size, im_encoder_hidden_size, batch_first=True)
        self.pr_encoder_lstm = nn.LSTM(pr_encoder_input_size, pr_encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, int(decoder_hidden_size/2), batch_first=True)

        self.decoder_fc_1 = nn.Linear(int(decoder_hidden_size/2), int(decoder_hidden_size/4))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/4), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoderIm(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.im_encoder_hidden_size)


    def initHiddenEncoderPR(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.pr_encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,int(self.decoder_hidden_size/2))


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)
        lstm_input_PR = torch.cat(PR, 1).view(image_s[0].size(0), 1, -1)

        encoder_output_images, im_encoder_hidden = self.im_encoder_lstm(lstm_input_features,  im_encoder_hidden )
        encoder_output_PR, pr_encoder_hidden = self.pr_encoder_lstm(lstm_input_PR,  pr_encoder_hidden )

        lstm_input_decoder = torch.cat((encoder_output_images, encoder_output_PR), 2).view(image_s[0].size(0), 1, -1)
        decoder_output, decoder_hidden = self.LSTM_decoder(lstm_input_decoder, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, im_encoder_hidden, pr_encoder_hidden, decoder_hidden


class CNN_LSTM_decoder_images_PR(nn.Module):
    def __init__(self,cuda = True, h_dim=2688, z_dim=1024, decoder_input_size = 1000, decoder_hidden_size = 1000,  output_size = 20, drop_par = 0.2):
        super(CNN_LSTM_decoder_images_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = 1
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

        self.decoder_lstm = nn.LSTM(decoder_input_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)
        self.dropout0 = nn.Dropout(p=drop_par)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = self.dropout0(outputs)
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = [ torch.cat((features[i], PR[i]), 1).view(image_s[0].size(0), 1, -1) for i in range(use_n_im)]
        lstm_input_features = torch.cat(lstm_input_features, 2).view(image_s[0].size(0), 1, -1)


        decoder_output, decoder_hidden = self.LSTM_decoder(lstm_input_features, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden
