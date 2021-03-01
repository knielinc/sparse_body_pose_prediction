import torch
from torch import nn
import numpy as np
from Helpers.NumpyTorchHelpers import to_numpy, to_torch
from Helpers.glow.generate_config_glow import generate_cfg
import copy
from Helpers.glow.models import Glow
from Helpers.glow.builder import build

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GLOWNET(nn.Module):
    def __init__(self, x_channels, cond_channels):
        super().__init__()

        hparams = generate_cfg()
        built = build(x_channels, cond_channels, hparams, True)

        self.glow = built["graph"]
        self.optim = built["optim"]
        self.lrschedule = built["lrschedule"]
        self.devices = built["devices"]
        self.data_device = built["data_device"]
        self.loaded_step = built["loaded_step"]

        self.scalar_log_gaps = hparams.Train.scalar_log_gap
        self.validation_log_gaps = hparams.Train.validation_log_gap
        self.plot_gaps = hparams.Train.plot_gap
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm

    def train_model(self, input, output, eval_input, eval_output, learning_rate, epochs, batch_size, stack_count):
        self.global_step = self.loaded_step
        batch_size = min(batch_size, input.shape[0])
        for epoch in range(epochs):
            print("epoch", epoch)

            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = input[i * batch_size: (i + 1) * batch_size, :]
                target_output = output[i * batch_size: (i + 1) * batch_size, :]

                # set to training state
                self.glow.train()

                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step,
                                             **self.lrschedule["args"])

                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()

                # dropout_prob = 0.7
                # nr_of_inputs = model_input.shape[2]
                # for row in range(stack_count):
                #     for curr_batch in range(model_input.shape[0]):
                #         dropout = np.random.rand(nr_of_inputs) < dropout_prob
                #         model_input[curr_batch,row*27:(row+1)*27,:] = model_input[curr_batch,row*27:(row+1)*27,:] * dropout
                x = to_torch(target_output)

                cond = to_torch(model_input)

                # init LSTM hidden
                if hasattr(self.glow, "module"):
                    self.glow.module.init_lstm_hidden()
                else:
                    self.glow.init_lstm_hidden()

                # at first time, initialize ActNorm
                if self.global_step == 0:
                    self.glow(x, cond if cond is not None else None)
                    # re-init LSTM hidden
                    if hasattr(self.glow, "module"):
                        self.glow.module.init_lstm_hidden()
                    else:
                        self.glow.init_lstm_hidden()

                # print("n_params: " + str(self.count_parameters(self.glow)))

                # forward phase
                z, nll = self.glow(x=x, cond=cond)

                # loss
                loss_generative = Glow.loss_generative(nll)

                loss = loss_generative

                # backward
                self.glow.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.glow.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.glow.parameters(), self.max_grad_norm)
                # step
                self.optim.step()

                # if self.global_step % self.validation_log_gaps == 0:
                #     # set to eval state
                #     self.graph.eval()
                #
                #     # Validation forward phase
                #     loss_val = 0
                #     n_batches = 0
                #     for ii, val_batch in enumerate(self.val_data_loader):
                #         for k in val_batch:
                #             val_batch[k] = val_batch[k].to(self.data_device)
                #
                #         with torch.no_grad():
                #
                #             # init LSTM hidden
                #             if hasattr(self.graph, "module"):
                #                 self.graph.module.init_lstm_hidden()
                #             else:
                #                 self.graph.init_lstm_hidden()
                #
                #             z_val, nll_val = self.graph(x=val_batch["x"], cond=val_batch["cond"])
                #
                #             # loss
                #             loss_val = loss_val + Glow.loss_generative(nll_val)
                #             n_batches = n_batches + 1
                #
                #     loss_val = loss_val / n_batches
                #     self.writer.add_scalar("val_loss/val_loss_generative", loss_val, self.global_step)
                loss_val = 0.0
                # global step
                self.global_step += 1

            print(
                f'Loss: {loss.item():.5f}/ Validation Loss: {loss_val:.5f} '
            )

    def predict(self, input, STACKCOUNT, x_len):

        batch_size = input.shape[0]
        shape_1 = self.glow.z_shape[1]
        shape_2 = self.glow.z_shape[2]

        predicted = []

        self.glow.z_shape = [batch_size, shape_1, shape_2]

        curr_input = input[None, 0, :, 0, None]

        eps_std = 1.0

        # Initialize the pose sequence with ground truth test data
        clipframes = input.shape[2]

        # Initialize the lstm hidden state
        if hasattr(self.glow, "module"):
            self.glow.module.init_lstm_hidden()
        else:
            self.glow.init_lstm_hidden()
        # Loop through control sequence and generate new data
        for i in range(1, clipframes):
            # sample from Moglow
            sampled = self.glow(z=None, cond=to_torch(curr_input), eps_std=eps_std, reverse=True)

            # update saved pose sequence
            next_input = np.roll(curr_input, -x_len, axis=1)
            last_elem_start_idx = (STACKCOUNT - 1) * x_len
            last_elem_end_idx = (STACKCOUNT) * x_len

            next_input[:, last_elem_start_idx:last_elem_end_idx, ...] = to_numpy(sampled)
            next_input[:, last_elem_end_idx:, ...] = input[:, last_elem_end_idx:, i, None]
            curr_input = next_input
            predicted.append(to_numpy(sampled[0, :, 0]))
        return np.array(predicted)


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
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output,
                    learning_rate, epochs, batch_size):
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

                if not eval_input is None:
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
                model_output[:, :3] = upper_prediction[curr_eval_idx, 6:9]  # hip should stay the same

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

class RNNNET(nn.Module):
    def __init__(self, input_size, hands_size, enc_dims, latent_dim, rnn_num_layers, rnn_hidden_size, dec_dims,
                 output_size):
        super(RNNNET, self).__init__()
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
        self.lower_body_encoder = nn.Sequential(*lower_body_modules)
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

        lower_latent = torch.empty(hands_input.shape[0], sequence_length, self.num_latent_dim).to(device)

        for sequence_idx in range(sequence_length):
            lower_latent[:, sequence_idx, :] = self.lower_body_encoder(latent_input[:, sequence_idx, :])

        return lower_latent

    def forward(self, lower_body, hands_input):
        latent = self.encode(lower_body, hands_input)
        # hands_input_latent = self.input_encoder(hands_input)
        h_0 = torch.zeros(self.rnn_num_layers, lower_body.size(0), self.rnn_hidden_size).to(device).float()
        rnn_out, _ = self.rnn(latent, h_0)
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
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output,
                    learning_rate, epochs, batch_size):
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

                if not eval_input is None:
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
                model_output[:, :3] = upper_prediction[curr_eval_idx, 6:9]  # hip should stay the same

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

class RNNNET2(nn.Module):
    def __init__(self, input_size, hands_size, enc_dims, latent_dim, rnn_num_layers, rnn_hidden_size, dec_dims,
                 output_size):
        super(RNNNET2, self).__init__()
        self.input_size = input_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.num_latent_dim = latent_dim
        self.do_init = True

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
        self.lower_body_encoder = nn.Sequential(*lower_body_modules)
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

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        self.do_init = True

    def encode(self, lower_body, hands_input):
        latent_input = torch.cat((lower_body, hands_input), 2)
        latent_input = self.dropout(latent_input)

        lower_latent = self.lower_body_encoder(latent_input)

        return lower_latent

    def forward(self, lower_body, hands_input):
        latent = self.encode(lower_body, hands_input)
        # hands_input_latent = self.input_encoder(hands_input)
        # h_0 = torch.zeros(self.rnn_num_layers, lower_body.size(0), self.rnn_hidden_size).to(device).float()

        if self.do_init:
            rnn_out, self.hidden = self.rnn(latent)
            self.do_init = False
        else:
            rnn_out, self.hidden = self.rnn(latent, self.hidden)

        # rnn_out, _ = self.rnn(latent, h_0)
        rnn_out = rnn_out[:, -1, :]  # only last output

        latent2 = torch.cat((rnn_out, hands_input[:, -1, :]), 1)
        dec_in = self.decoder_input(latent2)
        # print(to_numpy(log_var[0]))
        dec_out = self.decoder(dec_in)
        out = self.final_layer(dec_out)
        return out

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output,
                    learning_rate, epochs, batch_size):
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
                self.init_hidden()

                model_output = self(model_input, input_conditional)
                loss = criterian(model_output, model_target_output)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                if not eval_input is None:
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

    def predict(self, eval_lower, eval_cond, STACKCOUNT):

        x_len = 27
        batch_size = 1

        curr_input_lower = eval_lower[:, 0:1, :]
        curr_input_cond = eval_cond[:, 0:1, :]

        # Initialize the pose sequence with ground truth test data
        clipframes = eval_lower.shape[1]
        predicted = []

        # Initialize the lstm hidden state
        self.init_hidden()
        # Loop through control sequence and generate new data
        for i in range(1, clipframes):
            # sample from Moglow
            sampled = self(to_torch(curr_input_lower), to_torch(curr_input_cond))

            # update saved pose sequence
            next_input = np.roll(curr_input_lower, -x_len, axis=2)
            last_elem_start_idx = (STACKCOUNT - 1) * x_len
            last_elem_end_idx = (STACKCOUNT) * x_len

            next_input[:, :, last_elem_start_idx:last_elem_end_idx, ...] = to_numpy(sampled)
            curr_input_lower = next_input
            curr_input_cond = eval_cond[:, i:i+1, :]

            predicted.append(to_numpy(sampled[0]))
        return np.array(predicted)

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

        self.gan_loss_net = GANLOSSNET(input_size, output_size, 256, 16, learning_rate=0.001)

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
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, output, eval_input, eval_output, learning_rate, epochs, batch_size):
        criterion = self.get_criterion()
        optimizer = self.get_optimizer(learning_rate=learning_rate)

        maxloss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = input[i * batch_size: (i + 1) * batch_size, :]
                target_output = output[i * batch_size: (i + 1) * batch_size, :]

                # Forward pass
                model_output = self(to_torch(model_input))

                loss = criterion(model_output, to_torch(target_output))

                # if epoch > 100:
                #     loss += gan_loss / 4
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                with torch.no_grad():
                    if not eval_input is None:
                        self.eval()
                        model_eval_output = self(to_torch(eval_input))
                        self.train()
                        test_loss = criterion(model_eval_output, to_torch(eval_output), eval=True)
                        test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], total loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                test_maxloss = 0.0

    def predict(self, input):
        target_output = None
        input = to_torch(input)
        with torch.no_grad():
            self.eval()
            target_output = self(input)
        return target_output

class FFNetGanLoss(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNetGanLoss, self).__init__()
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

        self.gan_loss_net = GANLOSSNET(input_size, output_size, 256, 16, learning_rate=0.001)

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

    def get_gan_criterion(self):
        return self.gan_loss_net.train_model_one_epoch_and_get_loss

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, output, eval_input, eval_output, learning_rate, epochs, batch_size):
        criterion = self.get_gan_criterion()
        optimizer = self.get_optimizer(learning_rate=learning_rate)

        maxloss = 0.0
        max_gan_loss = 0.0
        max_mse_loss = 0.0
        max_smoothness_loss = 0.0
        max_smoothness_loss_real = 0.0
        max_disc_loss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = input[i * batch_size: (i + 1) * batch_size, :]
                target_output = output[i * batch_size: (i + 1) * batch_size, :]

                # Forward pass
                model_output = self(to_torch(model_input))

                model_input = model_input.reshape(-1,40,model_input.shape[1])
                model_output = model_output.reshape(-1,40,model_output.shape[1])
                target_output = target_output.reshape(-1,40,target_output.shape[1])

                gan_loss, smoothness_loss, smoothness_loss_real, mse_loss, discriminator_loss = criterion(model_input, model_output, to_torch(target_output))
                loss = mse_loss * 2
                if epoch > 20:
                    loss += gan_loss / 10 + smoothness_loss / 2

                # if epoch > 100:
                #     loss += gan_loss / 4
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                max_gan_loss =  np.fmax(max_gan_loss, gan_loss.item())
                max_mse_loss =  np.fmax(max_mse_loss, mse_loss.item())
                max_smoothness_loss =  np.fmax(max_smoothness_loss, smoothness_loss.item())
                max_smoothness_loss_real =  np.fmax(max_smoothness_loss_real, smoothness_loss_real.item())
                max_disc_loss =  np.fmax(max_disc_loss, discriminator_loss.item())

                with torch.no_grad():
                    if not eval_input is None:
                        self.eval()
                        model_eval_output = self(to_torch(eval_input))
                        self.train()
                        test_loss = criterion(model_eval_output, to_torch(eval_output), eval=True)
                        test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 1 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], total loss: {maxloss:.8f}, gan loss: {max_gan_loss:.8f}, smoothness loss: {max_smoothness_loss:.8f}, smoothness loss real: {max_smoothness_loss_real:.8f}, mse loss: {max_mse_loss:.8f}, D loss: {max_disc_loss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                max_gan_loss = 0.0
                max_mse_loss = 0.0
                max_smoothness_loss = 0.0
                max_smoothness_loss_real = 0.0
                max_disc_loss = 0.0
                test_maxloss = 0.0

    def predict(self, input):
        target_output = None
        input = to_torch(input)
        with torch.no_grad():
            self.eval()
            target_output = self(input)
        return target_output

class RNNNETGANLOSS(nn.Module):
    def __init__(self, input_size, hands_size, enc_dims, latent_dim, rnn_num_layers, rnn_hidden_size, dec_dims,
                 output_size):
        super(RNNNETGANLOSS, self).__init__()
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
        self.lower_body_encoder = nn.Sequential(*lower_body_modules)
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
        self.gan_loss_net = GANRNNLOSSNET(input_size, output_size, 256, 16, learning_rate=0.001)

    def get_gan_criterion(self):
        return self.gan_loss_net.train_model_one_epoch_and_get_loss

    def encode(self, lower_body, hands_input):
        sequence_length = lower_body.shape[1]
        zeroes = torch.zeros(hands_input.shape[0], sequence_length, hands_input.shape[1]).to(device).float()
        hands_input_ = hands_input.unsqueeze(1)
        hands_input_ = zeroes + hands_input_
        latent_input = torch.cat((lower_body, hands_input_), 2)
        latent_input = self.dropout(latent_input)

        lower_latent = torch.empty(hands_input.shape[0], sequence_length, self.num_latent_dim).to(device)

        for sequence_idx in range(sequence_length):
            lower_latent[:, sequence_idx, :] = self.lower_body_encoder(latent_input[:, sequence_idx, :])

        return lower_latent

    def forward(self, lower_body, hands_input):
        latent = self.encode(lower_body, hands_input)
        # hands_input_latent = self.input_encoder(hands_input)
        h_0 = torch.zeros(self.rnn_num_layers, lower_body.size(0), self.rnn_hidden_size).to(device).float()
        rnn_out, _ = self.rnn(latent, h_0)
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
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output,
                    learning_rate, epochs, batch_size):
        optimizer = self.get_optimizer(learning_rate)
        criterion = self.get_gan_criterion()

        maxloss = 0.0
        max_gan_loss = 0.0
        max_mse_loss = 0.0
        max_smoothness_loss = 0.0
        max_smoothness_loss_real = 0.0
        max_disc_loss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = to_torch(input[i * batch_size: (i + 1) * batch_size, :])
                input_conditional = to_torch(conditional_input[i * batch_size: (i + 1) * batch_size, :])
                model_target_output = to_torch(output[i * batch_size: (i + 1) * batch_size])
                # Forward pass
                model_output = self(model_input, input_conditional)
                gan_loss, smoothness_loss, smoothness_loss_real, mse_loss, discriminator_loss = criterion(model_input, model_output, to_torch(model_target_output))

                loss = gan_loss + smoothness_loss + mse_loss
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                max_gan_loss =  np.fmax(max_gan_loss, gan_loss.item())
                max_mse_loss =  np.fmax(max_mse_loss, mse_loss.item())
                max_smoothness_loss =  np.fmax(max_smoothness_loss, smoothness_loss.item())
                max_smoothness_loss_real =  np.fmax(max_smoothness_loss_real, smoothness_loss_real.item())
                max_disc_loss =  np.fmax(max_disc_loss, discriminator_loss.item())


                if not eval_input is None:
                    with torch.no_grad():
                        self.eval()
                        eval_input_feet = to_torch(eval_input)
                        eval_input_hands = to_torch(eval_conditional_input)

                        model_eval_output = self(eval_input_feet, eval_input_hands)

                        self.train()

                        test_loss = criterian(model_eval_output, to_torch(eval_output))
                        test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], total loss: {maxloss:.8f}, gan loss: {max_gan_loss:.8f}, smoothness loss: {max_smoothness_loss:.8f}, smoothness loss real: {max_smoothness_loss_real:.8f}, mse loss: {max_mse_loss:.8f}, D loss: {max_disc_loss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                max_gan_loss = 0.0
                max_mse_loss = 0.0
                max_smoothness_loss = 0.0
                max_smoothness_loss_real = 0.0
                max_disc_loss = 0.0
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
                model_output[:, :3] = upper_prediction[curr_eval_idx, 6:9]  # hip should stay the same

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


class GANLOSSNET(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_size, learning_rate):
        super(GANLOSSNET, self).__init__()
        output_size = output_size * 40

        self.input_size = output_size# + input_size
        # self.noise = TorchLayers.GaussianNoise()
        self.l1 = nn.Linear(output_size, hidden_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.l4 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate
        self.input_size = output_size
        # self.noise = TorchLayers.GaussianNoise()

    def forward(self, input, sample_motion):
        # concated_input = torch.cat((input, sample_motion), dim=1)
        flattened = torch.flatten(sample_motion, start_dim=1)
        h1 = self.l1(flattened)
        h1 = self.relu(h1)

        h2 = self.l2(h1)
        h2 = self.relu2(h2)

        h3 = self.l3(h2)
        h4 = self.relu3(h3)

        out = self.l4(h4)
        out = self.sigmoid(out)
        return out

    def latent(self, x):
        h1 = self.l1(x)
        # out = self.noise(out)
        h2 = self.relu(h1)
        latent = self.l2(h2)
        latent = self.relu2(latent)
        hidden, self.hidden = self.rnn(latent)
        return hidden

    def get_mse_criterion(self):
        return nn.MSELoss()

    def get_smoothness_criterion(self):
        return self.get_smoothness_loss

    def get_smoothness_tensor(self, frame_length):
        ones = torch.ones(frame_length - 1)
        diag_elems = torch.ones(frame_length) * -2
        diag_elems[0] += 1
        diag_elems[-1] += 1
        return (torch.diag(ones, 1) + torch.diag(ones, -1) + torch.diag(diag_elems)).to(device)

    def get_smoothness_loss(self, model_output):
        smoothness_mat = self.get_smoothness_tensor(model_output.shape[1])
        smoothness_criterion = torch.matmul(smoothness_mat, model_output)
        test_norm = torch.norm(smoothness_criterion, dim=1)

        return torch.mean(test_norm)

    def get_discr_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def get_adverserial_loss(self):
        return torch.nn.BCELoss()

    def train_model_one_epoch_and_get_loss(self, input, generated, real, eval=False):
        input = to_torch(input)
        generated = to_torch(generated)
        real = to_torch(real)

        optimizer_D = self.get_discr_optimizer(learning_rate=self.learning_rate)

        from torch.autograd import Variable
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        valid = Variable(Tensor(real.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(generated.size(0), 1).fill_(0.0), requires_grad=False)

        adverserial_loss = self.get_adverserial_loss()

        self.zero_grad()

        real_loss = adverserial_loss(self(input, real), valid)
        fake_loss = adverserial_loss(self(input, generated.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        if eval == False:
            d_loss.backward()
            optimizer_D.step()

        mse_criterion = self.get_mse_criterion()
        smoothness_criterion = self.get_smoothness_criterion()

        # gen_loss = adverserial_loss(self(generated), valid) + smoothness_criterion(generated) + mse_criterion(generated, real)

        return adverserial_loss(self(input, generated), valid), smoothness_criterion(generated), smoothness_criterion(real), mse_criterion(generated, real), d_loss

    def eval_hidden_diff(self, prediction, reference):
        mse_loss = nn.MSELoss()

        pred_hidden = self.latent(prediction)
        ref_hidden = self.latent(reference)

        loss = mse_loss(pred_hidden, ref_hidden)
        #
        # mse_criterion = self.get_mse_criterion()
        # smoothness_criterion = self.get_smoothness_criterion()
        # loss = mse_criterion(to_torch(prediction), to_torch(reference)) + smoothness_criterion(to_torch(prediction))
        return loss.item()

class GANRNNLOSSNET(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_size, learning_rate):
        super(GANRNNLOSSNET, self).__init__()
        self.input_size = output_size# + input_size
        # self.noise = TorchLayers.GaussianNoise()
        self.l1 = nn.Linear(output_size, hidden_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.l2 = nn.Linear(hidden_size, latent_size)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.rnn = nn.GRU(latent_size, latent_size, 1, batch_first=True)

        self.l3 = nn.Linear(latent_size, hidden_size)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.l4 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.learning_rate = learning_rate
        self.input_size = output_size
        # self.noise = TorchLayers.GaussianNoise()

    def forward(self, input, sample_motion):
        # concated_input = torch.cat((input, sample_motion), dim=1)
        h1 = self.l1(sample_motion)
        h1 = self.relu(h1)

        h2 = self.l2(h1)
        latent = self.relu2(h2)

        rnn_latent = self.rnn(latent)

        h3 = self.l3(rnn_latent)
        h4 = self.relu3(h3)

        out = self.l4(h4)
        out = self.sigmoid(out)
        return out

    def latent(self, x):
        h1 = self.l1(x)
        # out = self.noise(out)
        h2 = self.relu(h1)
        latent = self.l2(h2)
        latent = self.relu2(latent)
        hidden, self.hidden = self.rnn(latent)
        return hidden

    def get_mse_criterion(self):
        return nn.MSELoss()

    def get_smoothness_criterion(self):
        return self.get_smoothness_loss

    def get_smoothness_tensor(self, frame_length):
        ones = torch.ones(frame_length - 1)
        diag_elems = torch.ones(frame_length) * -2
        diag_elems[0] += 1
        diag_elems[-1] += 1
        return (torch.diag(ones, 1) + torch.diag(ones, -1) + torch.diag(diag_elems)).to(device)

    def get_smoothness_loss(self, model_output):
        smoothness_mat = self.get_smoothness_tensor(model_output.shape[1])
        smoothness_criterion = torch.matmul(smoothness_mat, model_output)
        test_norm = torch.norm(smoothness_criterion, dim=1)

        return torch.mean(test_norm)

    def get_discr_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def get_adverserial_loss(self):
        return torch.nn.BCELoss()

    def train_model_one_epoch_and_get_loss(self, input, generated, real, eval=False):
        input = to_torch(input)
        generated = to_torch(generated)
        real = to_torch(real)

        optimizer_D = self.get_discr_optimizer(learning_rate=self.learning_rate)

        from torch.autograd import Variable
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        valid = Variable(Tensor(real.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(generated.size(0), 1).fill_(0.0), requires_grad=False)

        adverserial_loss = self.get_adverserial_loss()

        self.zero_grad()

        real_loss = adverserial_loss(self(input, real), valid)
        fake_loss = adverserial_loss(self(input, generated.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        if eval == False:
            d_loss.backward()
            optimizer_D.step()

        mse_criterion = self.get_mse_criterion()
        smoothness_criterion = self.get_smoothness_criterion()

        # gen_loss = adverserial_loss(self(generated), valid) + smoothness_criterion(generated) + mse_criterion(generated, real)

        return adverserial_loss(self(input, generated), valid), smoothness_criterion(generated), smoothness_criterion(real), mse_criterion(generated, real), d_loss

    def eval_hidden_diff(self, prediction, reference):
        mse_loss = nn.MSELoss()

        pred_hidden = self.latent(prediction)
        ref_hidden = self.latent(reference)

        loss = mse_loss(pred_hidden, ref_hidden)
        #
        # mse_criterion = self.get_mse_criterion()
        # smoothness_criterion = self.get_smoothness_criterion()
        # loss = mse_criterion(to_torch(prediction), to_torch(reference)) + smoothness_criterion(to_torch(prediction))
        return loss.item()

class PERCEPTUALLOSSNET(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(PERCEPTUALLOSSNET, self).__init__()
        self.input_size = input_size
        # self.noise = TorchLayers.GaussianNoise()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, latent_size)
        self.relu2 = nn.ReLU()

        self.rnn = nn.GRU(latent_size, latent_size, 1, batch_first=True)

        self.l3 = nn.Linear(latent_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = self.l1(x)
        # out = self.noise(out)
        h2 = self.relu(h1)
        latent = self.l2(h2)
        latent = self.relu2(latent)

        hidden, self.hidden = self.rnn(latent)

        # h1 = self.dropout(h1)
        h3 = self.l3(hidden)
        h4 = self.relu3(h3)
        out = self.l4(h4)

        return out

    def latent(self, x):
        h1 = self.l1(x)
        # out = self.noise(out)
        h2 = self.relu(h1)
        latent = self.l2(h2)
        latent = self.relu2(latent)
        hidden, self.hidden = self.rnn(latent)
        return hidden

    def get_mse_criterion(self):
        return nn.MSELoss()

    def get_smoothness_criterion(self):
        return self.get_smoothness_loss

    def get_smoothness_tensor(self, frame_length):
        ones = torch.ones(frame_length - 1)
        diag_elems = torch.ones(frame_length) * -2
        diag_elems[0] += 1
        diag_elems[-1] += 1
        return (torch.diag(ones, 1) + torch.diag(ones, -1) + torch.diag(diag_elems)).to(device)

    def get_smoothness_loss(self, model_output):
        smoothness_mat = self.get_smoothness_tensor(model_output.shape[1])
        smoothness_criterion = torch.matmul(smoothness_mat, model_output)
        test_norm = torch.norm(smoothness_criterion, dim=1)

        return torch.mean(test_norm)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, eval_input, learning_rate, epochs, batch_size):
        mse_criterion = self.get_mse_criterion()
        smoothness_criterion = self.get_smoothness_criterion()
        optimizer = self.get_optimizer(learning_rate=learning_rate)

        maxloss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = input[i * batch_size: (i + 1) * batch_size, :]
                target_output = input[i * batch_size: (i + 1) * batch_size, :]

                # Forward pass
                model_output = self(to_torch(model_input))


                # smoothness_term
                loss = mse_criterion(model_output, to_torch(target_output)) + smoothness_criterion(model_output)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                with torch.no_grad():
                    if not eval_input is None:
                        self.eval()
                        model_eval_output = self(to_torch(eval_input))
                        self.train()
                        test_loss = mse_criterion(model_eval_output, to_torch(eval_input))
                        test_maxloss = np.fmax(test_maxloss, test_loss.item())

            if (epoch) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
                maxloss = 0.0
                test_maxloss = 0.0

    def eval_hidden_diff(self, prediction, reference):
        mse_loss = nn.MSELoss()

        pred_hidden = self.latent(prediction)
        ref_hidden = self.latent(reference)

        loss = mse_loss(pred_hidden, ref_hidden)
        #
        # mse_criterion = self.get_mse_criterion()
        # smoothness_criterion = self.get_smoothness_criterion()
        # loss = mse_criterion(to_torch(prediction), to_torch(reference)) + smoothness_criterion(to_torch(prediction))
        return loss.item()

class VAENET(nn.Module):
    def __init__(self, input_size, hands_size, enc_dims, latent_dim, dec_dims, output_size):
        super(VAENET, self).__init__()
        self.input_size = input_size

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

        self.lower_body_encoder_mu = nn.Sequential(*lower_body_modules)
        self.lower_body_encoder_log_var = copy.deepcopy(self.lower_body_encoder_mu)
        self.input_encoder = nn.Sequential(*input_modules)

        lower_body_modules = []

        self.decoder_input = nn.Linear(latent_dim + hands_size, dec_dims[0])

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
        lower_latent_mu = self.lower_body_encoder_mu(torch.hstack((lower_body, hands_input)))
        lower_latent_log_var = self.lower_body_encoder_log_var(torch.hstack((lower_body, hands_input)))

        return [lower_latent_mu, lower_latent_log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, lower_body, hands_input):
        mu, log_var = self.encode(lower_body, hands_input)
        reparametrized_latent = self.reparameterize(mu, log_var)
        # hands_input_latent = self.input_encoder(hands_input)
        latent = torch.hstack((reparametrized_latent, hands_input))
        dec_in = self.decoder_input(latent)
        dec_out = self.decoder(dec_in)
        out = self.final_layer(dec_out)
        return out

    def get_criterion(self):
        return nn.MSELoss()

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                amsgrad=False)

    def train_model(self, input, conditional_input, output, eval_input, eval_conditional_input, eval_output,
                    learning_rate, epochs, batch_size):
        optimizer = self.get_optimizer(learning_rate)
        criterian = self.get_criterion()

        maxloss = 0.0
        test_maxloss = 0.0

        for epoch in range(epochs):
            for i in range(int(np.floor(input.shape[0] / batch_size))):
                model_input = to_torch(input[i * batch_size: (i + 1) * batch_size, :])
                input_conditional = to_torch(conditional_input[i * batch_size: (i + 1) * batch_size, :])

                model_target_output = to_torch(output[i * batch_size: (i + 1) * batch_size, :])
                # Forward pass
                model_output = self(model_input, input_conditional)
                loss = criterian(model_output, model_target_output)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                maxloss = np.fmax(maxloss, loss.item())

                if not eval_input is None:
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
        vels_and_accels = input[STACKCOUNT - 1, -26:]

        new_input_feet = torch.flatten(curr_input_mat)

        lower_body_poses = None
        with torch.no_grad():
            self.eval()
            for curr_eval_idx in range(STACKCOUNT, input.shape[0]):
                model_output = self(new_input_feet, vels_and_accels)
                model_output[:3] = upper_prediction[curr_eval_idx, 6:9]  # hip should stay the same

                if lower_body_poses is None:
                    lower_body_poses = model_output
                else:
                    lower_body_poses = torch.vstack((lower_body_poses, model_output))

                curr_input_mat = torch.roll(curr_input_mat, -1, 0)
                vels_and_accels = input[curr_eval_idx, -26:]
                curr_input_mat[-1] = model_output
                curr_input_mat[-1][:3] = upper_prediction[curr_eval_idx, 6:9]

                new_input_feet = torch.flatten(curr_input_mat)

        upper_prediction[STACKCOUNT:-1, 6:15] = lower_body_poses[:, :9]
        upper_prediction[STACKCOUNT:-1, 21:27] = lower_body_poses[:, 9:]
        output = upper_prediction
        return output
