from Helpers import MocapImporter
import numpy as np
import scipy.interpolate as sciInterp
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
from glob import glob
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd

def resample_mocap_data(target_delta_t, in_delta_t, data):
    nr_data_points = data.shape[0]
    animation_duration = nr_data_points * in_delta_t
    in_axis = np.linspace(start=0, stop=animation_duration, num=nr_data_points)
    target_axis = np.linspace(start=0, stop=animation_duration,
                              num=np.floor(animation_duration / target_delta_t).astype(np.int))
    sampler = sciInterp.interp1d(in_axis, data, axis=0)
    return sampler(target_axis)


def reshape_for_cnn(data):
    data = data.reshape(data.shape[0], -1, 6)
    data = np.swapaxes(data, 1, 2)
    return data


def reshape_from_cnn(data):
    data = np.swapaxes(data, 1, 2)
    data = data.reshape(data.shape[0], -1)
    return data


def augment_dataset(data, nr_of_angles):
    # 1 -> 180 degree rotation
    # 2 -> 2x 120x rotation
    angle_step = 360.0 / float(nr_of_angles)

    output = data

    for i in range(1, nr_of_angles):
        y_rot_angle = angle_step * i
        rotmat = R.from_euler('y', y_rot_angle, degrees=True)
        rotated_data = rotmat.apply(data)
        output = np.vstack((output, rotated_data))
    return output


class ParalellMLPProcessor():
    def __init__(self, nr_of_timesteps_per_feature, target_delta_T, augment_rotation_number):
        self.nr_of_timesteps_per_feature = nr_of_timesteps_per_feature
        self.target_delta_t = target_delta_T
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.feet_inputs = np.array([])
        self.feet_outputs = np.array([])
        self.heads = np.array([])
        self.min = 1.0
        self.max = 1.0
        self.input_scaler = None
        self.output_scaler = None
        self.feet_input_scaler = None
        self.feet_output_scaler = None
        self.augment_rotation_number = augment_rotation_number

    def append_file(self, file_name):

        mocap_importer = MocapImporter.Importer(file_name)
        if not mocap_importer.is_usable():
            return
        print("imported file :" + file_name)

        global_positions = mocap_importer.zipped_global_positions
        joint_names = mocap_importer.joint_names
        frame_time = mocap_importer.frame_time
        bone_dependencies = mocap_importer.bone_dependencies

        resampled_global_pos = resample_mocap_data(in_delta_t=frame_time, target_delta_t=self.target_delta_t,
                                                   data=global_positions.reshape(global_positions.shape[0], -1))

        resampled_global_pos = resampled_global_pos.reshape(resampled_global_pos.shape[0], -1, 3)
        resampled_global_pos = augment_dataset(resampled_global_pos.reshape(-1, 3),
                                               self.augment_rotation_number).reshape(-1, resampled_global_pos.shape[1],
                                                                                     3)

        head_idx = joint_names.index('Head')
        l_hand_idx = joint_names.index('LeftHand')
        r_hand_idx = joint_names.index('RightHand')
        l_foot_idx = joint_names.index('LeftFoot')
        r_foot_idx = joint_names.index('RightFoot')
        hip_idx = joint_names.index('Hips')
        l_shoulder_idx = joint_names.index('LeftArm')
        r_shoulder_idx = joint_names.index('RightArm')
        l_elbow_idx = joint_names.index('LeftForeArm')
        r_elbow_idx = joint_names.index('RightForeArm')
        l_knee_idx = joint_names.index('LeftLeg')
        r_knee_idx = joint_names.index('RightLeg')

        number_of_limbs = resampled_global_pos.shape[1]
        heads = resampled_global_pos[:, [head_idx], :]
        resampled_global_pos -= resampled_global_pos[:, [head_idx] * number_of_limbs, :]

        ff_set = resampled_global_pos[:,
                 [l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx,
                  r_elbow_idx, l_knee_idx, r_knee_idx], :]
        hand_inputs = ff_set[:, [0, 1], :]
        skeletal_outputs = ff_set[:, [2, 3, 4, 5, 6, 7, 8, 9, 10], :]

        ae_set = resampled_global_pos[:,
                 [l_hand_idx, r_hand_idx, hip_idx, l_foot_idx, r_foot_idx, l_knee_idx, r_knee_idx], :]

        feet_inputs = ae_set.reshape(ae_set.shape[0], -1)
        feet_outputs = ae_set[:, [2, 3, 4, 5, 6], :].reshape(ae_set.shape[0], -1)

        hand_inputs = hand_inputs.reshape(hand_inputs.shape[0], -1)
        skeletal_outputs = skeletal_outputs.reshape(skeletal_outputs.shape[0], -1)
        heads = heads.reshape(heads.shape[0], -1)

        vels = np.diff(np.hstack((hand_inputs, heads)), axis=0)
        accels = np.diff(vels, axis=0)

        rolled_hand_inputs = hand_inputs  # np.hstack((inputs,np.roll(inputs,-1, axis=0)))
        rolled_feet_inputs = feet_inputs
        for i in range(1, self.nr_of_timesteps_per_feature):
            rolled_hand_inputs = np.hstack((rolled_hand_inputs, np.roll(hand_inputs, i * 1, axis=0)))
            rolled_feet_inputs = np.hstack((rolled_feet_inputs, np.roll(feet_inputs, i * 1, axis=0)))

        assert (self.nr_of_timesteps_per_feature >= 2)
        vels = vels[(self.nr_of_timesteps_per_feature - 1):, :]
        accels = accels[(self.nr_of_timesteps_per_feature - 2):, :]

        rolled_feet_inputs = rolled_feet_inputs[self.nr_of_timesteps_per_feature:-1, :]
        feet_outputs = feet_outputs[self.nr_of_timesteps_per_feature + 1:, :]

        rolled_hand_inputs = rolled_hand_inputs[self.nr_of_timesteps_per_feature:, :]
        skeletal_outputs = skeletal_outputs[self.nr_of_timesteps_per_feature:, :]
        heads = heads[self.nr_of_timesteps_per_feature:, :]

        rolled_hand_inputs = np.hstack((rolled_hand_inputs, vels, accels))
        rolled_feet_inputs = np.hstack((rolled_feet_inputs, vels[:-1, :], accels[:-1, :]))

        if self.inputs.shape[0] == 0:
            self.inputs = rolled_hand_inputs
            self.outputs = skeletal_outputs
            self.heads = heads
            self.feet_inputs = rolled_feet_inputs
            self.feet_outputs = feet_outputs
        else:
            self.inputs = np.vstack((self.inputs, rolled_hand_inputs))
            self.outputs = np.vstack((self.outputs, skeletal_outputs))
            self.heads = np.vstack((self.heads, heads))
            self.feet_inputs = np.vstack((self.feet_inputs, rolled_feet_inputs))
            self.feet_outputs = np.vstack((self.feet_outputs, feet_outputs))

        self.__calc_scaling_fac()

    def append_folder(self, folder, ignore_files=[]):
        folders = glob(folder)

        for path in folders:
            allfiles = [f for f in listdir(path) if isfile(join(path, f))]

            for file in allfiles:
                f = path + file
                if not f in ignore_files:
                    self.append_file(f)
                else:
                    print("ignored file: " + f)

    def append_subfolders(self, dir, ignore_files=[]):
        folders = glob(dir)

        for path in folders:
            allfolders = [f for f in listdir(path) if isdir(join(path, f))]

            for folder in allfolders:
                f = path + "/" + folder + "/"
                self.append_folder(f, ignore_files)

    def __calc_scaling_fac(self):
        self.min = np.minimum(self.inputs.min(), self.outputs.min())
        self.max = np.maximum(self.inputs.max(), self.outputs.max())

    def get_scaled_inputs(self, nr_of_angles=0):
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(self.inputs)
        return self.input_scaler.transform(self.inputs)

    def get_scaled_outputs(self):
        self.output_scaler = StandardScaler()
        self.output_scaler.fit(self.outputs)
        return self.output_scaler.transform(self.outputs)

    def get_scaled_feet_inputs(self, nr_of_angles=0):
        self.feet_input_scaler = StandardScaler()
        self.feet_input_scaler.fit(self.feet_inputs)
        return self.feet_input_scaler.transform(self.feet_inputs)

    def get_scaled_feet_outputs(self):
        self.feet_output_scaler = StandardScaler()
        self.feet_output_scaler.fit(self.feet_outputs)
        return self.feet_output_scaler.transform(self.feet_outputs)

    def scale_back_input(self, data):
        return self.input_scaler.inverse_transform(data)

    def scale_back_output(self, data):
        return self.output_scaler.inverse_transform(data)

    def scale_input(self, data):
        return self.input_scaler.transform(data)

    def scale_output(self, data):
        return self.output_scaler.transform(data)

    def add_heads(self, data):
        datacopy = data
        for i in range(data.shape[1]):
            datacopy[:, i, :] += self.heads
        return datacopy

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

    def save(self, target_dir):
        np.savez(target_dir,
                 inputs=self.inputs,
                 outputs=self.outputs,
                 feet_inputs=self.feet_inputs,
                 feet_outputs=self.feet_outputs,
                 nr_of_timesteps_per_feature=self.nr_of_timesteps_per_feature,
                 target_delta_t=self.target_delta_t,
                 heads=self.heads,
                 augment_rotation_number=self.augment_rotation_number)

    def load_np(self, source_dir):
        numpy_importer = np.load(source_dir)

        self.inputs = numpy_importer['inputs']
        self.outputs = numpy_importer['outputs']
        self.feet_inputs = numpy_importer['feet_inputs']
        self.feet_outputs = numpy_importer['feet_outputs']
        self.nr_of_timesteps_per_feature = numpy_importer['nr_of_timesteps_per_feature']
        self.target_delta_t = numpy_importer['target_delta_t']
        self.heads = numpy_importer['heads']
        self.augment_rotation_number = numpy_importer['augment_rotation_number']
