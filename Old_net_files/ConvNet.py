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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 6  # 28x28
hidden_size = 400
output_size = 27
num_epochs = 400
batch_size = 200
learning_rate = 0.0001
STACKCOUNT = 20
TARGET_FPS = 60.0

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS)
folders = glob("E:/Master/Sorted Movement/Basketball/")

for path in folders:
    allfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in allfiles:
        f = path + file
        if f == "E:/Master/Sorted Movement/Basketball/06_12.bvh":
            continue
        training_prep.append_file(f)
        print("imported file :" + f)

eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS)
eval_prep.append_file("E:/Master/Sorted Movement/Basketball/06_12.bvh")

train_input = training_prep.get_scaled_inputs()
train_output = training_prep.get_scaled_outputs()


eval_input = training_prep.scale_input(eval_prep.inputs)
eval_output = training_prep.scale_output(eval_prep.outputs)

train_input = DataPreprocessor.reshape_for_cnn(train_input)
eval_input = DataPreprocessor.reshape_for_cnn(eval_input)

train_input, train_output = shuffle(train_input, train_output, random_state=42)


input_size = train_input.shape[1]
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, in_channels, kernel_size, in_length, num_classes, hidden_channels=6):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.c1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(((in_length - kernel_size + 1) - kernel_size + 1) * hidden_channels, hidden_size)
        self.relu3 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.c1(x)
        #out = self.relu(out)
        out = self.c2(out)
        out = self.relu2(out)
        out = self.flatten(out)
        out = self.l1(out)
        out = self.relu3(out)
        out = self.l2(out)
        out = self.relu4(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, 5, STACKCOUNT, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999), eps = 1e-08, weight_decay=0, amsgrad=False)

# Train the model
n_total_steps = int(train_input.shape[0] / batch_size)
maxloss = 0.0
test_maxloss = 0.0
train_input  = torch.tensor(train_input).to(device).float()
train_output = torch.tensor(train_output).to(device).float()
eval_input   = torch.tensor(eval_input).to(device).float()
eval_output  = torch.tensor(eval_output).to(device).float()

for epoch in range(num_epochs):
    for i in range(int(np.floor(train_input.shape[0] / batch_size))):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]

        input = train_input[i * batch_size : (i + 1) * batch_size, :]
        target_output = train_output[i * batch_size: (i + 1) * batch_size, :]

        # Forward pass
        model_output = model(input)
        loss = criterion(model_output, target_output)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        maxloss = np.fmax(maxloss, loss.item())


        with torch.no_grad():
            model_eval_output = model(eval_input)
            test_loss = criterion(model_eval_output, eval_output)
            test_maxloss = np.fmax(test_maxloss, test_loss.item())

        # if (i + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Train Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
        #     test_maxloss = 0.0
        #     maxloss = 0.0

    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {maxloss:.8f}, Test Loss: {test_maxloss:.8f}')
        maxloss = 0.0
        test_maxloss = 0.0


with torch.no_grad():
    target_output = model(eval_input)


# "l_hand_idx, r_hand_idx, l_elbow_idx, r_elbow_idx, hip_idx, l_foot_idx, r_foot_idx
#[l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx, r_elbow_idx, l_knee_idx, r_knee_idx]
bone_dependencies = [[0,7],[1,8],[2,4],[3,4],[4,-1],[5,9],[6,10],[7,2],[8,3],[9,4],[10,4]]
bone_dependencies = np.array(bone_dependencies)


global_positions = np.hstack((training_prep.scale_back_input(DataPreprocessor.reshape_from_cnn(eval_input.detach().cpu().numpy()))[:,:6], training_prep.scale_back_output(target_output.detach().cpu().numpy())))
# global_positions = np.hstack((eval_input, eval_output))
global_positions = global_positions.reshape(global_positions.shape[0], -1, 3)
global_positions = eval_prep.add_heads(global_positions)
# training_prep.scale_back(global_positions)

if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, ['']*40, bone_dependencies, 1.0/TARGET_FPS)
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

