import torch
from torch import nn
import numpy as np
from Helpers.NumpyTorchHelpers import to_numpy, to_torch
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNVAENET(nn.Module):
    def __init__(self, input_size, hands_size, enc_dims, latent_dim, rnn_num_layers, rnn_hidden_size, dec_dims,
                 output_size):
        super(RNNVAENET, self).__init__()
        self.input_size = input_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.num_latent_dim = latent_dim

        input_size_ = input_size + hands_size
        input_size_hands_ = hands_size
        lower_body_modules = []
        input_modules = []

        for h_dim in enc_dims:
            lower_body_modules.append(
                nn.Sequential(
                    nn.Linear(input_size_, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            input_size_ = h_dim

        lower_body_modules.append(
            nn.Sequential(
                nn.Linear(input_size_, out_features=latent_dim),
                # nn.BatchNorm1d(h_dim),
                nn.LeakyReLU())
        )

        for h_dim in enc_dims:
            input_modules.append(
                nn.Sequential(
                    nn.Linear(input_size_hands_, out_features=h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            input_size_hands_ = h_dim

        input_modules.append(
            nn.Sequential(
                nn.Linear(input_size_hands_, out_features=latent_dim),
                # nn.BatchNorm1d(h_dim),
                nn.LeakyReLU())
        )
        self.dropout = nn.Dropout(p=0.7)
        self.lower_body_encoder_mu = nn.Sequential(*lower_body_modules)
        self.lower_body_encoder_log_var = copy.deepcopy(self.lower_body_encoder_mu)
        self.input_encoder = nn.Sequential(*input_modules)

        # print(self.input_encoder)
        lower_body_modules = []
        # shape = [batchsize, sequence_length, input_size]
        self.rnn = nn.GRU(latent_dim, rnn_hidden_size, rnn_num_layers, batch_first=True)

        self.decoder_input = nn.Linear(rnn_hidden_size + hands_size, dec_dims[0])

        for i in range(len(dec_dims) - 1):
            lower_body_modules.append(
                nn.Sequential(
                    nn.Linear(dec_dims[i], out_features=dec_dims[i + 1]),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*lower_body_modules)

        self.final_layer = nn.Sequential(
            nn.Linear(dec_dims[-1], dec_dims[-1]),
            # nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(dec_dims[-1], output_size))

    def encode(self, lower_body, hands_input):
        sequence_length = lower_body.shape[1]
        zeroes = torch.zeros(hands_input.shape[0], sequence_length, hands_input.shape[1]).to(device).float()
        hands_input_ = hands_input.unsqueeze(1)
        hands_input_ = zeroes + hands_input_
        latent_input = torch.cat((lower_body, hands_input_), 2)
        latent_input = self.dropout(latent_input)

        lower_latent_mu = torch.empty(hands_input.shape[0], sequence_length, self.num_latent_dim).to(device)
        lower_latent_log_var = torch.empty(hands_input.shape[0], sequence_length, self.num_latent_dim).to(device)

        for sequence_idx in range(sequence_length):
            lower_latent_mu[:, sequence_idx, :] = self.lower_body_encoder_mu(latent_input[:, sequence_idx, :])
            lower_latent_log_var[:, sequence_idx, :] = self.lower_body_encoder_log_var(latent_input[:, sequence_idx, :])

        return [lower_latent_mu, lower_latent_log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, lower_body, hands_input):
        mu, log_var = self.encode(lower_body, hands_input)
        reparametrized_latent = self.reparameterize(mu, log_var)
        # hands_input_latent = self.input_encoder(hands_input)
        h_0 = torch.zeros(self.rnn_num_layers, lower_body.size(0), self.rnn_hidden_size).to(device).float()
        rnn_out, _ = self.rnn(reparametrized_latent, h_0)
        rnn_out = rnn_out[:, -1, :]  # only last output

        latent = torch.hstack((rnn_out, hands_input))
        dec_in = self.decoder_input(latent)
        # print(to_numpy(log_var[0]))
        dec_out = self.decoder(dec_in)
        out = self.final_layer(dec_out)
        return out

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output, learning_rate, epochs, batch_size):
        optimizer = self.get_optimizer(learning_rate)
        criterian = self.get_criterion()

        maxloss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = to_torch(input[i * batch_size: (i + 1) * batch_size, :])
                input_conditional = to_torch(conditional_input[i * batch_size: (i + 1) * batch_size, :])
                model_target_output = to_torch(output[i * batch_size: (i + 1) * batch_size])
                # Forward pass
                model_output = self(model_input, input_conditional)
                loss = criterian(model_output, model_target_output)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                with torch.no_grad():
                    self.eval()
                    eval_input_feet = to_torch(eval_input)
                    eval_input_hands = to_torch(eval_conditional_input)

                    model_eval_output = self(eval_input_feet, eval_input_hands)

                    self.train()

                    test_loss = criterian(model_eval_output, to_torch(eval_output))
                    test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                test_maxloss = 0.0

    def predict(self, input, upper_prediction, STACKCOUNT):
        input = to_torch(input)
        upper_prediction = to_torch(upper_prediction)

        curr_input_mat = torch.hstack((upper_prediction[:STACKCOUNT, 6:15], upper_prediction[:STACKCOUNT, 21:27]))
        vels_and_accels = input[STACKCOUNT - 1, -26:].unsqueeze(0)

        new_input_feet = torch.flip(curr_input_mat, [0]).unsqueeze(0)

        lower_body_poses = None
        with torch.no_grad():
            self.eval()
            for curr_eval_idx in range(STACKCOUNT, input.shape[0]):
                model_output = self(new_input_feet, vels_and_accels)
                model_output[:, :3] = upper_prediction[curr_eval_idx, 6:9] #hip should stay the same

                if lower_body_poses is None:
                    lower_body_poses = model_output
                else:
                    lower_body_poses = torch.vstack((lower_body_poses, model_output))

                curr_input_mat = torch.roll(curr_input_mat, -1, 0)
                vels_and_accels = input[curr_eval_idx, -26:].unsqueeze(0)
                curr_input_mat[-1] = model_output

                new_input_feet = curr_input_mat.unsqueeze(0)

        upper_prediction[STACKCOUNT:-1, 6:15] = lower_body_poses[:, :9]
        upper_prediction[STACKCOUNT:-1, 21:27] = lower_body_poses[:, 9:]
        output = upper_prediction
        return output


class FFNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNet, self).__init__()
        self.input_size = input_size
        # self.noise = TorchLayers.GaussianNoise()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = self.l1(x)
        # out = self.noise(out)
        h1 = self.relu(h1)
        h1 = self.l2(h1)
        h1 = self.relu2(h1)
        h1 = self.dropout(h1)
        h1 = self.l3(h1)
        h1 = self.relu3(h1)
        out = self.l4(h1)

        return out

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    def train_model(self, input, output, eval_input, eval_output, learning_rate, epochs, batch_size):
        criterion = self.get_criterion()
        optimizer = self.get_optimizer(learning_rate= learning_rate)

        maxloss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = input[i * batch_size: (i + 1) * batch_size, :]
                target_output = output[i * batch_size: (i + 1) * batch_size, :]

                # Forward pass
                model_output = self(to_torch(model_input))
                loss = criterion(model_output, to_torch(target_output))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                with torch.no_grad():
                    self.eval()
                    model_eval_output = self(to_torch(eval_input))
                    self.train()
                    test_loss = criterion(model_eval_output, to_torch(eval_output))
                    test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                test_maxloss = 0.0

    def predict(self, input):
        target_output = None
        input = to_torch(input)
        with torch.no_grad():
            self.eval()
            target_output = self(input)
        return target_output
