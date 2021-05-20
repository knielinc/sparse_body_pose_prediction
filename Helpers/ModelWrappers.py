from Helpers.Models import FFNet, RNNVAENET, GLOWNET, VAENET, RNNNET, RNNNET2, PERCEPTUALLOSSNET, FFNetGanLoss, RNNGan
from Helpers import Animator
from Helpers import DataPreprocessor
from Helpers import StatsPrinter as sp
from Helpers.NumpyTorchHelpers import to_numpy, to_torch
from sklearn.utils import shuffle
from Helpers import AnimationStacker

import numpy as np
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class model_wrapper():
    def __init__(self, train_prep):
        self.train_prep = train_prep
        self.last_inference_time = None


    def train(self):
        pass

    def predict(self, eval_prep):
        self.eval_prep = eval_prep
        pass

    def save_prediction(self, name, perceptual_loss_net):
        pass

    def export_anim_unity(self, localpositions, name):
        motion = []
        #pos
        motion.append(self.eval_prep.heads)
        #rot
        motion.append(self.eval_prep.heading_dirs)

        #lhand
        lhand = self.eval_prep.inputs[:, :3]
        motion.append(lhand)
        #rhand
        rhand = self.eval_prep.inputs[:, 3:6]
        motion.append(rhand)
        #rhanddir
        for i in range(int(localpositions.shape[1]/3)):
            idx = i*3
            motion.append(localpositions[:,idx:idx+3])

        file_name = name.split('/')[-1]
        name = name.replace(file_name, "unity_motion_export/" + file_name)
        name += ".ume"
        open(name, 'w').close()
        with open(name, 'wb') as f:
            for row in motion:
                np.savetxt(f, [row.reshape(-1)], delimiter=',')

    def stats_printer(self, local_positions, reference_positions, name, loss_net):
        file_name = name.split('/')[-1]
        target_path = name.replace(file_name, "stats/")

        name = name.replace(file_name, "stats/" + file_name)
        inputs = self.eval_prep.inputs
        cond_input = np.concatenate((inputs[:,:6],inputs[:,-20:]),1)
        local_positions_        = local_positions.reshape(local_positions.shape[0], -1, 3)
        reference_positions_    = reference_positions.reshape(reference_positions.shape[0], -1, 3)

        sp.print_stats(local_positions_, reference_positions_,
                       ["l_shoulder", "r_shoulder", "hip", "l_foot", "r_foot", "l_elbow", "r_elbow",
                        "l_knee", "r_knee"], name, self.eval_prep.target_delta_t, loss_net, cond_input=cond_input, target_path=target_path)

    def save_anim(self, global_positions, reference_positions, bone_dependencies, rotations, name):
        file_name = name.split('/')[-1]
        name = name.replace(file_name, "videos_out/" + file_name)

        anim = Animator.MocapAnimator2(global_positions, [''] * 40, bone_dependencies, self.train_prep.target_delta_t,
                                       heading_dirs=rotations,
                                       name="trained.mp4")
        anim.animation()
        reference_anim = Animator.MocapAnimator2(reference_positions, [''] * 40, bone_dependencies,
                                                 self.train_prep.target_delta_t,
                                                 heading_dirs=rotations,
                                                 name="reference.mp4")
        reference_anim.animation()
        AnimationStacker.concatenate_animations("trained.mp4", "reference.mp4", name + ".mp4")


class glow_wrapper(model_wrapper):
    def __init__(self, train_prep):
        super(glow_wrapper, self).__init__(train_prep)

    def train(self, num_epochs, batch_size):
        super(glow_wrapper, self).train()

        training_prep = self.train_prep

        train_cond = training_prep.get_scaled_inputs_glow()
        train_cond = train_cond.reshape(-1, training_prep.total_seq_length, train_cond.shape[1]).transpose((0, 2, 1))
        train_x = training_prep.get_scaled_outputs_glow()
        train_x = train_x.reshape(-1, training_prep.total_seq_length, train_x.shape[1]).transpose((0, 2, 1))

        x_channels = train_x.shape[1]
        cond_channels = train_cond.shape[1]

        self.glow_model = GLOWNET(x_channels, cond_channels).to(device)

        print("\n\nGLOW Train:\n")
        self.glow_model.train_model(input=train_cond,
                                    output=train_x,
                                    eval_input=None,
                                    eval_output=None,
                                    learning_rate=None,
                                    epochs=num_epochs,
                                    batch_size=batch_size,
                                    stack_count=training_prep.nr_of_timesteps_per_feature,
                                    train_prep=training_prep)

    def predict(self, eval_prep):
        super(glow_wrapper, self).predict(eval_prep)
        training_prep = self.train_prep

        eval_cond = training_prep.scale_inputs_glow(eval_prep.glow_inputs, False)  # .scale_input(eval_prep.inputs)
        eval_cond = eval_cond.reshape(-1, eval_cond.shape[0], eval_cond.shape[1]).transpose((0, 2, 1))
        eval_x = training_prep.scale_outputs_glow(eval_prep.glow_outputs, False)  # scale_output(eval_prep.outputs)
        eval_x = eval_x.reshape(-1, eval_x.shape[0], eval_x.shape[1]).transpose((0, 2, 1))

        cond_count = 26
        x_len = eval_x.shape[1]
        before_time = time.perf_counter()
        self.final_outputs = self.glow_model.predict(eval_cond, training_prep.nr_of_timesteps_per_feature, x_len)
        after_time = time.perf_counter()
        inference_time = after_time - before_time
        self.last_inference_time = inference_time

    def save_prediction(self, name, perceptual_loss_net):
        super(glow_wrapper, self).save_prediction(name, perceptual_loss_net)

        training_prep = self.train_prep
        eval_prep = self.eval_prep
        final_outputs = self.final_outputs


        eval_input = training_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = training_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        idx1 = (eval_input.shape[0]) % eval_prep.total_seq_length
        idx2 = idx1 + final_outputs.shape[0]
        eval_input = eval_input[idx1:idx2, :]
        eval_output = eval_output[idx1:idx2, :]

        local_positions = to_numpy(final_outputs)

        bone_dependencies, global_positions, rotations = self.eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                       local_positions,
                                                                                                       training_prep,
                                                                                                       start_idx=idx1,
                                                                                                       end_idx=idx2)
        bone_dependencies, reference_positions, rotations = self.eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                          eval_output,
                                                                                                          training_prep,
                                                                                                          start_idx=idx1,
                                                                                                          end_idx=idx2)


        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)

class rnn_wrapper(model_wrapper):
    def __init__(self, train_prep):
        super(rnn_wrapper, self).__init__(train_prep)

    def train(self, upper_num_epochs, lower_num_epochs, batch_size, learning_rate):
        super(rnn_wrapper, self).train()
        training_prep = self.train_prep
        train_hands_input = training_prep.get_scaled_inputs()
        train_hands_output = training_prep.get_scaled_outputs()

        # train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)

        hands_input_size = 26
        rnn_num_layers = 6
        rnn_hidden_size = 8
        hidden_size = 350
        output_size = 27

        upper_body_input_size = train_hands_input.shape[1]

        self.ff_model = FFNet(upper_body_input_size, hidden_size, output_size).to(device)

        print("\n\nUpper Body Train FF :\n")
        self.ff_model.train_model(input=train_hands_input,
                                  output=train_hands_output,
                                  eval_input=None,
                                  eval_output=None,
                                  learning_rate=learning_rate,
                                  epochs=upper_num_epochs,
                                  batch_size=batch_size)

        print("\n\nLower Body Train RNN:\n")

        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature

        train_feet_input = training_prep.get_scaled_feet_inputs()
        train_feet_output = training_prep.get_scaled_feet_outputs()

        # train_feet_input, train_feet_output = shuffle(train_feet_input, train_feet_output, random_state=42)

        feet_input_size = int((train_feet_input.shape[1] - hands_input_size) / STACKCOUNT)
        feet_output_size = train_feet_output.shape[1]

        self.rnnvae_model = RNNNET(feet_input_size, hands_input_size, [64, 32, 16], 16, rnn_num_layers,
                                      rnn_hidden_size,
                                      [400, 400, 256, 64], feet_output_size).to(device)

        train_feet_lower_input = train_feet_input[:, :-26].reshape(train_feet_input.shape[0], STACKCOUNT, -1)
        train_feet_lower_input = np.flip(train_feet_lower_input, 1)
        train_feet_lower_conditional_input = train_feet_input[:, -26:]

        self.rnnvae_model.train_model(input=train_feet_lower_input,
                                      conditional_input=train_feet_lower_conditional_input,
                                      output=train_feet_output,
                                      eval_input=None,
                                      eval_conditional_input=None,
                                      eval_output=None,
                                      learning_rate=learning_rate,
                                      epochs=lower_num_epochs,
                                      batch_size=batch_size)

    def predict(self, eval_prep):
        super(rnn_wrapper, self).predict(eval_prep)

        eval_feet_input = self.train_prep.scale_feet_input(eval_prep.feet_inputs)
        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)

        ff_outputs = self.ff_model.predict(eval_input)
        self.final_outputs = self.rnnvae_model.predict(eval_feet_input, ff_outputs, STACKCOUNT)

    def save_prediction(self, name, perceptual_loss_net):
        super(rnn_wrapper, self).save_prediction(name, perceptual_loss_net)
        eval_prep = self.eval_prep
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = self.train_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        local_positions = to_numpy(self.final_outputs)


        bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                  local_positions,
                                                                                                  self.train_prep)
        _, reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output,
                                                                                     self.train_prep)
        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)

class rnn_wrapper_2(model_wrapper):
    def __init__(self, train_prep):
        super(rnn_wrapper_2, self).__init__(train_prep)

    def train(self, lower_num_epochs, batch_size, learning_rate):
        super(rnn_wrapper_2, self).train()
        training_prep = self.train_prep
        train_hands_input = training_prep.get_scaled_inputs()
        train_hands_output = training_prep.get_scaled_outputs()

        # train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)

        hands_input_size = 26
        rnn_num_layers = 6
        rnn_hidden_size = 8

        print("\n\nTrain RNN2:\n")


        first_part_end = training_prep.nr_of_timesteps_per_feature * 27
        train_inputs_lower = training_prep.get_scaled_inputs_glow()
        train_inputs_lower = train_inputs_lower.reshape(-1, training_prep.total_seq_length, train_inputs_lower.shape[1])
        train_lower = train_inputs_lower[:, :, :first_part_end]
        train_cond = train_inputs_lower[:, :, first_part_end:]
        train_x = training_prep.get_scaled_outputs_glow()
        train_x = train_x.reshape(-1, training_prep.total_seq_length, train_x.shape[1])

        rnn_input_size = train_lower.shape[2]
        rnn_cond_size = train_cond.shape[2]
        rnn_output_size = train_x.shape[2]

        self.rnn_model = RNNNET2(rnn_input_size, rnn_cond_size, [64, 32, 27], 27, rnn_num_layers, rnn_hidden_size,
                                   [400, 256, 64], rnn_output_size).to(device)

        batch_size /= training_prep.nr_of_timesteps_per_feature
        # batch_size = 1
        batch_size = int(batch_size)
        self.rnn_model.train_model(input=train_lower,
                                   conditional_input=train_cond,
                                   output=train_x,
                                   eval_input=None,
                                   eval_conditional_input=None,
                                   eval_output=None,
                                   learning_rate=learning_rate,
                                   epochs=lower_num_epochs,
                                   batch_size=batch_size,
                                   train_prep=training_prep)

    def predict(self, eval_prep):
        super(rnn_wrapper_2, self).predict(eval_prep)
        training_prep = self.train_prep
        first_part_end = training_prep.nr_of_timesteps_per_feature * 27

        eval_inputs_lower = training_prep.scale_inputs_glow(eval_prep.glow_inputs, False)  # .scale_input(eval_prep.inputs)
        eval_inputs_lower = eval_inputs_lower.reshape(1, -1, eval_inputs_lower.shape[1])
        eval_lower = eval_inputs_lower[:, :, :first_part_end]
        eval_cond = eval_inputs_lower[:, :, first_part_end:]


        before_time = time.perf_counter()
        self.final_outputs = self.rnn_model.predict(eval_lower, eval_cond, training_prep.nr_of_timesteps_per_feature)
        after_time = time.perf_counter()
        inference_time = after_time - before_time
        self.last_inference_time = inference_time

    def save_prediction(self, name, perceptual_loss_net):
        super(rnn_wrapper_2, self).save_prediction(name, perceptual_loss_net)

        training_prep   = self.train_prep
        eval_prep       = self.eval_prep
        final_outputs   = self.final_outputs

        eval_input  = training_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = training_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        idx1 = (eval_input.shape[0]) % eval_prep.total_seq_length
        idx2 = idx1 + final_outputs.shape[0]
        eval_input = eval_input[idx1:idx2, :]
        eval_output = eval_output[idx1:idx2, :]
        self.final_outputs = self.final_outputs.squeeze(1)

        local_positions = to_numpy(self.final_outputs)

        bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                  local_positions,
                                                                                                  self.train_prep,
                                                                                                  start_idx=idx1,
                                                                                                  end_idx=idx2)
        _, reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output,
                                                                                     self.train_prep,
                                                                                                  start_idx=idx1,
                                                                                                  end_idx=idx2)
        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)

    def save_model(self, path):
        torch.save(self.rnn_model.state_dict(), path)

    def load_model(self, path):
        rnn_num_layers = 6
        rnn_hidden_size = 8
        rnn_input_size = 405
        rnn_cond_size = 432
        rnn_output_size = 27

        self.rnn_model = RNNNET2(rnn_input_size, rnn_cond_size, [64, 32, 27], 27, rnn_num_layers, rnn_hidden_size,
                                   [400, 256, 64], rnn_output_size).to(device)

        self.rnn_model.load_state_dict(torch.load(path))
        # self.rnn_model.eval()


class ff_wrapper(model_wrapper):
    def __init__(self, train_prep):
        super(ff_wrapper, self).__init__(train_prep)

    def train(self, num_epochs, batch_size, learning_rate):
        super(ff_wrapper, self).train()
        training_prep = self.train_prep
        train_hands_input = training_prep.get_scaled_inputs()
        train_hands_output = training_prep.get_scaled_outputs()

        # train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)

        hidden_size = 350
        output_size = 27

        upper_body_input_size = train_hands_input.shape[1]

        self.ff_model = FFNetGanLoss(upper_body_input_size, hidden_size, output_size).to(device)

        print("\n\nFull Body Train FF :\n")
        self.ff_model.train_model(input=train_hands_input,
                                  output=train_hands_output,
                                  eval_input=None,
                                  eval_output=None,
                                  learning_rate=learning_rate,
                                  epochs=num_epochs,
                                  batch_size=batch_size,
                                  train_prep=training_prep)

    def predict(self, eval_prep):
        super(ff_wrapper, self).predict(eval_prep)

        eval_feet_input = self.train_prep.scale_feet_input(eval_prep.feet_inputs)
        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)

        before_time = time.perf_counter()
        self.final_outputs = self.ff_model.predict(eval_input)
        after_time = time.perf_counter()
        inference_time = after_time - before_time
        self.last_inference_time = inference_time

    def save_prediction(self, name, perceptual_loss_net):
        super(ff_wrapper, self).save_prediction(name, perceptual_loss_net)
        eval_prep = self.eval_prep
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = self.train_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        local_positions = to_numpy(self.final_outputs)

        bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                  local_positions,
                                                                                                  self.train_prep)
        _, reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output,
                                                                                     self.train_prep)
        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)


class gan_wrapper(model_wrapper):
    def __init__(self, train_prep):
        super(gan_wrapper, self).__init__(train_prep)

    def train(self, num_epochs, batch_size, learning_rate):
        super(gan_wrapper, self).train()
        training_prep = self.train_prep
        train_hands_input = training_prep.get_scaled_inputs()
        train_hands_output = training_prep.get_scaled_outputs()

        hidden_size = 150
        output_size = 27
        latent_size = 15

        upper_body_input_size = train_hands_input.shape[1]

        self.gan_model = RNNGan(upper_body_input_size, hidden_size, latent_size, output_size).to(device)

        print("\n\nGAN TRAIN :\n")
        self.gan_model.train_model(input=train_hands_input,
                                  output=train_hands_output,
                                  eval_input=None,
                                  eval_output=None,
                                  learning_rate=learning_rate,
                                  epochs=num_epochs,
                                  batch_size=batch_size,
                                   train_prep=training_prep)

    def predict(self, eval_prep):
        super(gan_wrapper, self).predict(eval_prep)

        eval_feet_input = self.train_prep.scale_feet_input(eval_prep.feet_inputs)
        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)

        self.final_outputs = self.gan_model.predict(eval_input)

    def save_prediction(self, name, perceptual_loss_net):
        super(gan_wrapper, self).save_prediction(name, perceptual_loss_net)
        eval_prep = self.eval_prep
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = self.train_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        local_positions = to_numpy(self.final_outputs)

        bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                  local_positions,
                                                                                                  self.train_prep)
        _, reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output,
                                                                                     self.train_prep)

        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)

    def get_loss(self, input, prediction, reference):
        seq_len = self.gan_model.gan_loss_net.seq_len

        predicted_  = prediction.reshape(-1, seq_len, 27)
        labeled_    = reference.reshape(-1, seq_len, 27)
        input_ = input.reshape(-1, seq_len, input.shape[1])

        return self.gan_model.gan_loss_net.eval_hidden_diff(to_torch(input_), to_torch(predicted_), to_torch(labeled_))

    def get_other_loss(self, input, prediction):
        seq_len = self.gan_model.gan_loss_net.seq_len

        predicted_  = prediction.reshape(-1, seq_len, 27)
        input_ = input.reshape(-1, seq_len, input.shape[1])

        return self.gan_model.gan_loss_net.forward(to_torch(input_), to_torch(predicted_))

class vae_wrapper(model_wrapper):
    def __init__(self, train_prep):
        super(vae_wrapper, self).__init__(train_prep)

    def train(self, upper_num_epochs, lower_num_epochs, batch_size, learning_rate):
        super(vae_wrapper, self).train()
        training_prep = self.train_prep
        train_hands_input = training_prep.get_scaled_inputs()
        train_hands_output = training_prep.get_scaled_outputs()

        # train_hands_input, train_hands_output = shuffle(train_hands_input, train_hands_output, random_state=42)

        hands_input_size = 26
        rnn_num_layers = 6
        rnn_hidden_size = 8
        hidden_size = 350
        output_size = 27

        upper_body_input_size = train_hands_input.shape[1]

        self.ff_model = FFNet(upper_body_input_size, hidden_size, output_size).to(device)

        print("\n\nUpper Body Train FF :\n")
        self.ff_model.train_model(input=train_hands_input,
                                  output=train_hands_output,
                                  eval_input=None,
                                  eval_output=None,
                                  learning_rate=learning_rate,
                                  epochs=upper_num_epochs,
                                  batch_size=batch_size)

        print("\n\nLower Body Train vae:\n")

        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature

        train_feet_input = training_prep.get_scaled_feet_inputs()
        train_feet_output = training_prep.get_scaled_feet_outputs()

        # train_feet_input, train_feet_output = shuffle(train_feet_input, train_feet_output, random_state=42)

        feet_input_size = int((train_feet_input.shape[1] - hands_input_size))
        feet_output_size = train_feet_output.shape[1]

        self.vae_model = VAENET(feet_input_size,
                                hands_input_size,
                                [64, 32, 16], 6, [400, 400, 256, 64],
                                feet_output_size).to(device)

        train_feet_lower_input = train_feet_input[:, :-26]
        train_feet_lower_conditional_input = train_feet_input[:, -26:]

        self.vae_model.train_model(input=train_feet_lower_input,
                                   conditional_input=train_feet_lower_conditional_input,
                                   output=train_feet_output,
                                   eval_input=None,
                                   eval_conditional_input=None,
                                   eval_output=None,
                                   learning_rate=learning_rate,
                                   epochs=lower_num_epochs,
                                   batch_size=batch_size)

    def predict(self, eval_prep):
        super(vae_wrapper, self).predict(eval_prep)

        eval_feet_input = self.train_prep.scale_feet_input(eval_prep.feet_inputs)
        STACKCOUNT = self.train_prep.nr_of_timesteps_per_feature
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)

        ff_outputs = self.ff_model.predict(eval_input)
        self.final_outputs = self.vae_model.predict(eval_feet_input, ff_outputs, STACKCOUNT)

    def save_prediction(self, name, perceptual_loss_net):
        super(vae_wrapper, self).save_prediction(name, perceptual_loss_net)
        eval_prep = self.eval_prep
        eval_input = self.train_prep.scale_input(eval_prep.inputs)  # .scale_input(eval_prep.inputs)
        eval_output = self.train_prep.scale_output(eval_prep.outputs)  # scale_output(eval_prep.outputs)

        local_positions = to_numpy(self.final_outputs)

        bone_dependencies, global_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input,
                                                                                                  local_positions,
                                                                                                  self.train_prep)
        _, reference_positions, rotations = eval_prep.get_global_pos_from_prediction(eval_input, eval_output,
                                                                                     self.train_prep)
        self.export_anim_unity(self.train_prep.scale_back_output(local_positions), name)
        self.stats_printer(self.train_prep.scale_back_output(local_positions), eval_prep.outputs, name, perceptual_loss_net)
        self.save_anim(global_positions, reference_positions, bone_dependencies, rotations, name)


class ae_perceptual_loss():
    def __init__(self, train_prep):
        self.train_prep = train_prep

    def train(self, epochs, batch_size, learning_rate):
        input_size = 27
        hidden_size = 32
        latent_size = 16
        output_size = 27

        self.perceptual_loss_net = PERCEPTUALLOSSNET(input_size, hidden_size, latent_size, output_size).to(device)

        train_x = self.train_prep.get_scaled_outputs_glow()
        train_x = train_x.reshape(-1, self.train_prep.total_seq_length, train_x.shape[1])

        self.perceptual_loss_net.train_model(train_x, None, learning_rate, epochs, batch_size)

    def get_loss(self, predicted, labeled):
        predicted_  = predicted.reshape(1, predicted.shape[0], 27)
        labeled_    = labeled.reshape(1, labeled.shape[0], 27)

        return self.perceptual_loss_net.eval_hidden_diff(to_torch(predicted_), to_torch(labeled_))

    def save_prediction(self, name):
        pass