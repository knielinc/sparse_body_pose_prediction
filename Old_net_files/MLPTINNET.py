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
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 6  # 28x28
hidden_size = 350
output_size = 27
num_epochs = 40
batch_size = 60000
learning_rate = 0.0001
STACKCOUNT = 10
TARGET_FPS = 20.0
#walking : 41_05

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)

training_prep.load_np("C:/Users/cknie/Desktop/convertedMocapData_bvh/all_bvh_data.npz")


eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 1)
eval_prep.append_file("C:/Users/cknie/Desktop/convertedMocapData_bvh/Basketball/06_12.npz")

train_input = training_prep.get_scaled_inputs()
train_output = training_prep.get_scaled_outputs()

train_input, train_output = shuffle(train_input, train_output, random_state=42)

eval_input = training_prep.scale_input(eval_prep.inputs)
eval_output = training_prep.scale_output(eval_prep.outputs)

input_size = train_input.shape[1]
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        #self.noise = TorchLayers.GaussianNoise()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropo = nn.Dropout(p=0.4)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()

        self.enc_1 = nn.Linear(num_classes, num_classes)
        self.e_relu1 = nn.ReLU()
        self.enc_2 = nn.Linear(num_classes, num_classes//2)
        self.e_relu2 = nn.ReLU()

        self.l5 = nn.Linear(hidden_size + num_classes//2, hidden_size)
        self.relu4 = nn.ReLU()
        self.l6 = nn.Linear(hidden_size, num_classes)


    def forward(self, x, prev_o):
        h1 = self.l1(x)
        #out = self.noise(out)
        h1 = self.relu(h1)
        h1 = self.l2(h1)
        h1 = self.relu2(h1)
        #out = self.dropo(out)
        h1 = self.l3(h1)
        h1 = self.relu3(h1)

        h2 = self.enc_1(prev_o)
        h2 = self.e_relu1(h2)
        h2 = self.enc_2(h2)
        h2 = self.e_relu2(h2)

        out = torch.hstack((h1, h2))
        out = self.l5(out)
        out = self.relu4(out)
        out = self.l6(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps = 1e-08, weight_decay=0, amsgrad=False)

# Train the model
n_total_steps = int(train_input.shape[0] / batch_size)
maxloss = 0.0
test_maxloss = 0.0
previous_outputs = np.roll(train_output, 1, axis=0)
previous_outputs[0, :] = train_output[0 , :]
previous_outputs = torch.tensor(previous_outputs).to(device).float();
previous_output2 = torch.tensor(train_output[0, :]).to(device).float();

train_input  = torch.tensor(train_input).to(device).float()
train_output = torch.tensor(train_output).to(device).float()
eval_input   = torch.tensor(eval_input).to(device).float()
eval_output  = torch.tensor(eval_output).to(device).float()

for epoch in range(num_epochs):
    for i in range(int(np.floor(train_input.shape[0] / batch_size))):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]

        input = train_input[i * batch_size : (i + 1) * batch_size, :]
        previous_output = previous_outputs[i * batch_size: (i + 1) * batch_size, :]
        target_output = train_output[i * batch_size: (i + 1) * batch_size, :]

        # Forward pass
        model_output = model(input, previous_output)
        loss = criterion(model_output, target_output)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        maxloss = np.fmax(maxloss, loss.item())

        model_output.detach_()
        previous_outputs[(i * batch_size + 1): 1 + (i + 1) * batch_size, :] = model_output


    if (epoch) % 1 == 0:
        with torch.no_grad():
            model.eval()
            prev_out = previous_output2.unsqueeze(0)
            for j in range(eval_input.shape[0]):

                model_eval_output = model(eval_input[j].unsqueeze(0), prev_out)
                prev_out = model_eval_output

                test_loss = criterion(model_eval_output, eval_output[j])
                test_maxloss = np.fmax(test_maxloss, test_loss.item())
            model.train()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
        maxloss = 0.0
        test_maxloss = 0.0

target_output = previous_output2.unsqueeze(0)

with torch.no_grad():
    model.eval()
    prev_out = previous_output2.unsqueeze(0)
    for j in range(eval_input.shape[0]):
        model_eval_output = model(eval_input[j].unsqueeze(0), prev_out)
        prev_out = model_eval_output
        target_output = torch.vstack((target_output, prev_out))
target_output = target_output[1:, :]



# "l_hand_idx, r_hand_idx, l_elbow_idx, r_elbow_idx, hip_idx, l_foot_idx, r_foot_idx
#[l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx, r_elbow_idx, l_knee_idx, r_knee_idx]
bone_dependencies = [[0,7],[1,8],[2,4],[3,4],[4,-1],[5,9],[6,10],[7,2],[8,3],[9,4],[10,4]]
bone_dependencies = np.array(bone_dependencies)


global_positions = np.hstack((training_prep.scale_back_input(eval_input.detach().cpu().numpy())[:,:6], training_prep.scale_back_output(target_output.detach().cpu().numpy())))
# global_positions = np.hstack((eval_input, eval_output))
global_positions = global_positions.reshape(global_positions.shape[0], -1, 3)
global_positions = eval_prep.add_heads(global_positions)
# training_prep.scale_back(global_positions)

if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, ['']*40, bone_dependencies, 1.0/TARGET_FPS,  write_to_file=True)
    anim.animation()


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for input, output in test_loader:
#         input = input.reshape(-1, 28 * 28).to(device)
#         output = output.to(device)
#         outputs = model(input)
#         # max returns (value ,index)
#         predicted = torch.round(10 * outputs.data).T
#         n_samples += output.size(0)
#         n_correct += (predicted == output).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')


