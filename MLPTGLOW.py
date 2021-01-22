import numpy as np
import torch.onnx
from Helpers import Animator
from Helpers import DataPreprocessor
from sklearn.utils import shuffle
from Helpers import StatsPrinter as sp
from Helpers.NumpyTorchHelpers import to_numpy, to_torch
from Helpers.Models import FFNet, RNNVAENET, GLOWNET


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
upper_body_input_size = 6  # 28x28
hidden_size = 350
output_size = 27
num_epochs = 1
batch_size = 20
learning_rate = 0.0001
STACKCOUNT = 10
TARGET_FPS = 20.0
# walking : 41_05
eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 1)
eval_prep.append_file("C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/17_03.npz")

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
training_prep.append_folder("C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/", ["C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/17_03.npz"])

train_cond = training_prep.get_scaled_inputs_glow()
train_cond = train_cond.reshape(-1, training_prep.total_seq_length, train_cond.shape[1]).transpose((0, 2, 1))
train_x = training_prep.get_scaled_outputs_glow()
train_x = train_x.reshape(-1, training_prep.total_seq_length, train_x.shape[1]).transpose((0, 2, 1))

# train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)
eval_cond = training_prep.scale_inputs_glow(eval_prep.glow_inputs, False)  # .scale_input(eval_prep.inputs)
eval_cond = eval_cond.reshape(-1, eval_cond.shape[0], eval_cond.shape[1]).transpose((0, 2, 1))
eval_x = training_prep.scale_outputs_glow(eval_prep.glow_outputs, False)  # scale_output(eval_prep.outputs)
eval_x = eval_x.reshape(-1, eval_x.shape[0], eval_x.shape[1]).transpose((0, 2, 1))

x_channels = train_x.shape[1]
cond_channels = train_cond.shape[1]

glow_model = GLOWNET(x_channels, cond_channels).to(device)

print("\n\nUpper Body Train:\n")
glow_model.train_model(input=train_cond,
                     output=train_x,
                     eval_input=eval_cond,
                     eval_output=eval_x,
                     learning_rate=learning_rate,
                     epochs=num_epochs,
                     batch_size=batch_size)

cond_count = 26
x_len = eval_x.shape[1]
final_outputs = glow_model.predict(eval_cond[:, :, :500], STACKCOUNT, x_len)
#
eval_input = training_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
eval_output = training_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

idx1 = (eval_input.shape[0]) % eval_prep.total_seq_length
idx2 = idx1 + 499
eval_input = eval_input[idx1:idx2, :]
eval_output = eval_output[idx1:idx2, :]

bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, to_numpy(final_outputs), training_prep, start_idx=idx1, end_idx=idx1+499)
bone_dependencies , reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output, training_prep, start_idx=idx1, end_idx=idx2)

MOTIONTYPE = "Boxing"
# sp.print_stats(global_positions, reference_positions, ["l_hand", "r_hand", "l_shoulder", "r_shoulder", "hip", "l_foot", "r_foot", "l_elbow", "r_elbow", "l_knee", "r_knee"], MOTIONTYPE)

if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, [''] * 40, bone_dependencies, 1.0 / TARGET_FPS, heading_dirs=rotations, name="trained.avi")
    anim.animation()
    reference_anim = Animator.MocapAnimator(reference_positions, [''] * 40, bone_dependencies, 1.0 / TARGET_FPS, heading_dirs=erotations, name="reference.avi")
    reference_anim.animation()
    from Helpers import AnimationStacker
    AnimationStacker.concatenate_animations("trained.avi", "reference.avi", MOTIONTYPE+".mp4")
