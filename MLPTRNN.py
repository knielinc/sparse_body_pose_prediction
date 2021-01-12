import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx
from glob import glob
from os import listdir
from os.path import isfile, join

from Helpers import MocapImporter
from Helpers import Animator
from Helpers import DataPreprocessor
from sklearn.utils import shuffle
from Helpers import TorchLayers
import copy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
upper_body_input_size = 6  # 28x28
hidden_size = 350
output_size = 27
num_epochs = 150
batch_size = 6000
learning_rate = 0.0001
STACKCOUNT = 10
TARGET_FPS = 20.0
# walking : 41_05


training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)

training_prep.append_folder("E:/Master/Sorted Movement/Walking/", ["E:/Master/Sorted Movement/Walking/15_14.bvh"])
training_prep.append_folder("E:/Master/Sorted Movement/Basketball/")
training_prep.append_folder("E:/Master/Sorted Movement/Boxing/")
training_prep.append_folder("E:/Master/Sorted Movement/Soccer/")
training_prep.append_folder("E:/Master/Sorted Movement/Football/")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/MPI_HDM05")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/KIT", ["C:/Users/cknie/Desktop/convertedMocapData/KIT/dance_waltz12_poses.npz"])

# training_prep.load_np('C:/Users/cknie/Desktop/convertedMocapData/all_BMLhandball_data.npz')

# training_prep.load_np('E:/Master/Sorted Movement/all_2.npz')
training_prep.save('E:/Master/Sorted Movement/Walking_Basketball_Boxing_Soccer_Football.npz')

# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/ACCAD")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/BMLhandball")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/BMLmovi")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/DFaust_67")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/EKUT")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/Eyes_Japan_Dataset")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/HumanEva")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/Kit")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/MPI_HDM05")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/MPI_Limits")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/MPI_mosh")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/SFU")
# training_prep.append_subfolders("C:/Users/cknie/Desktop/convertedMocapData/TotalCapture")


eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 1)
eval_prep.append_file("E:/Master/Sorted Movement/Walking/15_14.bvh")

train_hands_input = training_prep.get_scaled_inputs()
train_hands_output = training_prep.get_scaled_outputs()

train_feet_input = training_prep.get_scaled_feet_inputs()
train_feet_output = training_prep.get_scaled_feet_outputs()

train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)
train_feet_input, train_feet_output = shuffle(train_feet_input, train_feet_output, random_state=42)

# train_feet_input = np.hstack((train_feet_input[:,0:15], train_feet_input[:,-24:]))

eval_input = training_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
eval_output = training_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

eval_feet_input = training_prep.scale_feet_input(eval_prep.feet_inputs)
eval_feet_output = training_prep.scale_feet_output(eval_prep.feet_outputs)

# eval_feet_input = np.hstack((eval_feet_input[:,0:15], eval_feet_input[:,-24:]))


hands_input_size = 26
rnn_num_layers = 6
rnn_hidden_size = 8

upper_body_input_size = train_hands_input.shape[1]

# Fully connected neural network with one hidden layer

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
        self.dropout = nn.Dropout(p=0.9)
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
        # latent_input = self.dropout(latent_input)

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


ff_model = FFNet(upper_body_input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ff_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                             amsgrad=False)

to_numpy = lambda x: x.detach().cpu().numpy()
to_torch = lambda x: torch.tensor(x).to(device).float()

# Train the model
n_total_steps = int(train_hands_input.shape[0] / batch_size)
maxloss = 0.0
test_maxloss = 0.0
train_hands_input = to_torch(train_hands_input)
train_hands_output = to_torch(train_hands_output)

eval_input = to_torch(eval_input)
eval_output = to_torch(eval_output)

for epoch in range(num_epochs):
    for i in range(int(np.floor(train_hands_input.shape[0] / batch_size))):
        input = train_hands_input[i * batch_size: (i + 1) * batch_size, :]
        target_output = train_hands_output[i * batch_size: (i + 1) * batch_size, :]

        # Forward pass
        model_output = ff_model(input)
        loss = criterion(model_output, target_output)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        maxloss = np.fmax(maxloss, loss.item())

        with torch.no_grad():
            ff_model.eval()
            model_eval_output = ff_model(eval_input)
            ff_model.train()
            test_loss = criterion(model_eval_output, eval_output)
            test_maxloss = np.fmax(test_maxloss, test_loss.item())

    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
        maxloss = 0.0
        test_maxloss = 0.0

with torch.no_grad():
    ff_model.eval()
    target_output = ff_model(eval_input)

print("\n\nFEET:")
maxloss = 0.0
test_maxloss = 0.0
num_epochs = 90

feet_input_size = int((train_feet_input.shape[1] - hands_input_size) / STACKCOUNT)
feet_output_size = train_feet_output.shape[1]

train_feet_input = to_torch(train_feet_input)
train_feet_output = to_torch(train_feet_output)

eval_feet_input = to_torch(eval_feet_input)
eval_feet_output = to_torch(eval_feet_output)

rnnvae_model = RNNVAENET(feet_input_size, hands_input_size, [64, 32, 16], 16, rnn_num_layers, rnn_hidden_size,
                         [400, 400, 256, 64], feet_output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ff_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                             amsgrad=False)

feet_criterion = nn.MSELoss()
feet_optimizer = torch.optim.Adam(rnnvae_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=0,
                                  amsgrad=False)

for epoch in range(num_epochs):
    for i in range(int(np.floor(train_feet_input.shape[0] / batch_size))):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]

        input = train_feet_input[i * batch_size: (i + 1) * batch_size, :].to(device).float()
        target_output_feet = train_feet_output[i * batch_size: (i + 1) * batch_size, :].to(device).float()
        input_feet = input[:, :-26].reshape(input.shape[0], STACKCOUNT, -1)
        input_feet = torch.flip(input_feet, [1])
        input_hands = input[:, -26:]

        # Forward pass
        model_output = rnnvae_model(input_feet, input_hands)
        loss = feet_criterion(model_output, target_output_feet)

        # Backward and optimize
        feet_optimizer.zero_grad()
        loss.backward()
        feet_optimizer.step()
        maxloss = np.fmax(maxloss, loss.item())

        with torch.no_grad():
            rnnvae_model.eval()
            eval_input_feet = eval_feet_input[:, :-26].reshape(eval_feet_input.shape[0], STACKCOUNT, -1)
            eval_input_feet = torch.flip(eval_input_feet, [1])
            eval_input_hands = eval_feet_input[:, -26:]
            model_eval_output = rnnvae_model(eval_input_feet, eval_input_hands)
            rnnvae_model.train()
            test_loss = criterion(model_eval_output, eval_feet_output)
            test_maxloss = np.fmax(test_maxloss, test_loss.item())

    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
        maxloss = 0.0
        test_maxloss = 0.0


STACKCOUNT = 10
curr_input_mat = torch.hstack((target_output[:STACKCOUNT, 6:15], target_output[:STACKCOUNT, 21:27]))
vels_and_accels = eval_feet_input[STACKCOUNT - 1, -26:].unsqueeze(0)
# new_input_feet = curr_input_mat.unsqueeze(0)
new_input_feet = torch.flip(curr_input_mat, [0]).unsqueeze(0)

lower_body_poses = None
with torch.no_grad():
    rnnvae_model.eval()
    for curr_eval_idx in range(STACKCOUNT, eval_feet_input.shape[0]):
        model_output = rnnvae_model(new_input_feet, vels_and_accels)
        model_output[:,:3] = target_output[curr_eval_idx, 6:9]

        if lower_body_poses is None:
            lower_body_poses = model_output
        else:
            lower_body_poses = torch.vstack((lower_body_poses, model_output))

        curr_input_mat = torch.roll(curr_input_mat, -1, 0)
        vels_and_accels = eval_feet_input[curr_eval_idx, -26:].unsqueeze(0)
        curr_input_mat[-1] = model_output

        #new_input_feet = torch.flip(curr_input_mat.unsqueeze(0), [0])
        new_input_feet = curr_input_mat.unsqueeze(0)

target_output[STACKCOUNT:-1, 6:15] = lower_body_poses[:, :9]
target_output[STACKCOUNT:-1, 21:27] = lower_body_poses[:, 9:]

# 'Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg''Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'
# "l_hand_idx, r_hand_idx, l_elbow_idx, r_elbow_idx, hip_idx, l_foot_idx, r_foot_idx
#         -         -             0             1             2         3          4            5           6            7            8
#         0         1             2             3             4         5          6            7           8            9            10
#                            'LeftArm',     'RightArm',     'Hips' 'LeftFoot' 'RightFoot' 'LeftForeArm' 'RightForeArm' 'LeftLeg','RightLeg'
# [l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx, r_elbow_idx, l_knee_idx, r_knee_idx]
bone_dependencies = [[0, 7], [1, 8], [2, 4], [3, 4], [4, -1], [5, 9], [6, 10], [7, 2], [8, 3], [9, 4], [10, 4]]
bone_dependencies = np.array(bone_dependencies)

global_positions = np.hstack((training_prep.scale_back_input(to_numpy(eval_input))[:, :6],
                              training_prep.scale_back_output(to_numpy(target_output))))
# global_positions = np.hstack((eval_input, eval_output))
global_positions = global_positions.reshape(global_positions.shape[0], -1, 3)
global_positions = eval_prep.rotate_back(global_positions)
global_positions = eval_prep.add_heads(global_positions)
# training_prep.scale_back(global_positions)

if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, [''] * 40, bone_dependencies, 1.0 / TARGET_FPS, heading_dirs=eval_prep.heading_dirs[-global_positions.shape[0]:])
    anim.animation()
