import numpy as np
import torch.onnx
from Helpers import Animator
from Helpers import DataPreprocessor
from sklearn.utils import shuffle
from Helpers import StatsPrinter as sp
from Helpers.NumpyTorchHelpers import to_numpy, to_torch
from Helpers.Models import FFNet, RNNVAENET


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
eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 1)
eval_prep.append_file("C:/Users/cknie/Desktop/convertedMocapData/Kit/575/MarcusS_AdrianM05_poses.npz")

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
training_prep.append_file("C:/Users/cknie/Desktop/convertedMocapData/Kit/575/MarcusS_AdrianM05_poses.npz")

# training_prep.append_folder("E:/Master/Sorted Movement/Boxing/", ["E:/Master/Sorted Movement/Boxing/15_13.bvh"])
# training_prep.append_folder("E:/Master/Sorted Movement/Walking/", ["E:/Master/Sorted Movement/Walking/15_14.bvh"])

# training_prep.append_folder("E:/Master/Sorted Movement/Walking/")
# training_prep.append_folder("E:/Master/Sorted Movement/Basketball/")
# training_prep.append_folder("E:/Master/Sorted Movement/Basketball/", ["E:/Master/Sorted Movement/Basketball/06_13.bvh"])
# training_prep.append_folder("E:/Master/Sorted Movement/Boxing/")
# training_prep.append_folder("E:/Master/Sorted Movement/Soccer/")
# training_prep.append_folder("E:/Master/Sorted Movement/Football/")
# training_prep.append_subfolders("E:/Master/Converted Mocap//MPI_HDM05")
# training_prep.append_subfolders("E:/Master/Converted Mocap//KIT", ["E:/Master/Converted Mocap//KIT/dance_waltz12_poses.npz"])

# training_prep.load_np('C:/Users/cknie/Desktop/convertedMocapData/all_BMLhandball_data.npz')

# training_prep.load_np('E:/Master/Sorted Movement/Walking_Basketball_Boxing_Soccer_Football.npz')
# training_prep.save('E:/Master/Sorted Movement/Walking_Basketball_Boxing_Soccer_Football.npz')

# training_prep.append_subfolders("E:/Master/Converted Mocap/ACCAD")
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLhandball")
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLmovi")
# training_prep.append_subfolders("E:/Master/Converted Mocap/DFaust_67")
# training_prep.append_subfolders("E:/Master/Converted Mocap/EKUT")
# training_prep.append_subfolders("E:/Master/Converted Mocap/Eyes_Japan_Dataset")
# training_prep.append_subfolders("E:/Master/Converted Mocap/HumanEva")
# training_prep.append_subfolders("E:/Master/Converted Mocap/Kit")
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_HDM05")
# training_prep.append_subfolders("E:/Master/Converted Moca/MPI_Limits")
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_mosh")
# training_prep.append_subfolders("E:/Master/Converted Mocap/SFU")
# training_prep.append_subfolders("E:/Master/Converted Mocap/TotalCapture")

# training_prep.load_np("C:/Users/cknie/Desktop/convertedMocapData/combined.npz")


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
rnn_hidden_size = 8

upper_body_input_size = train_hands_input.shape[1]

ff_model = FFNet(upper_body_input_size, hidden_size, output_size).to(device)

print("\n\nUpper Body Train:\n")
ff_model.train_model(input=train_hands_input,
                     output=train_hands_output,
                     eval_input=eval_input,
                     eval_output=eval_output,
                     learning_rate=learning_rate,
                     epochs=num_epochs,
                     batch_size=batch_size)

print("\n\nLower Body Train:\n")

num_epochs = 90

feet_input_size = int((train_feet_input.shape[1] - hands_input_size) / STACKCOUNT)
feet_output_size = train_feet_output.shape[1]

rnnvae_model = RNNVAENET(feet_input_size, hands_input_size, [64, 32, 16], 16, rnn_num_layers, rnn_hidden_size,
                         [400, 400, 256, 64], feet_output_size).to(device)

train_feet_lower_input = train_feet_input[:, :-26].reshape(train_feet_input.shape[0], STACKCOUNT, -1)
train_feet_lower_input = np.flip(train_feet_lower_input, 1)
train_feet_lower_conditional_input = train_feet_input[:, -26:]

eval_feet_lower_input = eval_feet_input[:, :-26].reshape(train_feet_input.shape[0], STACKCOUNT, -1)
eval_feet_lower_input = np.flip(eval_feet_lower_input, 1)
eval_feet_conditional_input = eval_feet_input[:, -26:]

rnnvae_model.train_model(input=train_feet_lower_input,
                         conditional_input=train_feet_lower_conditional_input,
                         output=train_feet_output,
                         eval_input=eval_feet_lower_input,
                         eval_conditional_input=eval_feet_conditional_input,
                         eval_output=eval_output,
                         learning_rate=learning_rate,
                         epochs=num_epochs,
                         batch_size=batch_size)

ff_outputs = ff_model.predict(eval_input)
final_outputs = rnnvae_model.predict(eval_feet_input, ff_outputs, STACKCOUNT)

bone_dependencies, global_positions = eval_prep.get_global_pos_from_prediction(eval_input, to_numpy(final_outputs), training_prep)
_ , reference_positions = eval_prep.get_global_pos_from_prediction(eval_input, eval_output, training_prep)

MOTIONTYPE = "Boxing"
sp.print_stats(global_positions, reference_positions, ["l_hand", "r_hand", "l_shoulder", "r_shoulder", "hip", "l_foot", "r_foot", "l_elbow", "r_elbow", "l_knee", "r_knee"], MOTIONTYPE)

if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, [''] * 40, bone_dependencies, 1.0 / TARGET_FPS, heading_dirs=eval_prep.heading_dirs[-global_positions.shape[0]:], name="trained.avi")
    anim.animation()
    reference_anim = Animator.MocapAnimator(reference_positions, [''] * 40, bone_dependencies, 1.0 / TARGET_FPS, heading_dirs=eval_prep.heading_dirs[-reference_positions.shape[0]:], name="reference.avi")
    reference_anim.animation()
    from Helpers import AnimationStacker
    AnimationStacker.concatenate_animations("trained.avi", "reference.avi", MOTIONTYPE+".mp4")
