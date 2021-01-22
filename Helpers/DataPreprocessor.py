from Helpers import MocapImporter
import numpy as np
import scipy.interpolate as sciInterp
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
from glob import glob
from os import listdir
from os.path import isfile, join, isdir
import math
TWO_PI = (np.pi * 2)

def fit_pos_vector_from_names(pos_scaler, mat, names, inverse=False):
    mat_ = mat.copy()

    lengths = []
    sums = None
    for name in names:
        curr_length = pos_scaler[name].scale_.shape[0]
        lengths.append(curr_length)
        if sums is None:
            sums = [curr_length]
        else:
            sums.append(sums[-1] + curr_length)
    for loop_idx in range(mat.shape[1] // sums[-1]):
        for i in range(lengths.__len__()):
            start_index = sums[i] - lengths[i] + loop_idx * sums[-1]
            name = names[i]
            idx1 = start_index
            idx2 = start_index + lengths[i]
            if(inverse):
                mat_[:, idx1:idx2] = pos_scaler[name].inverse_transform(mat[:, idx1:idx2])
            else:
                mat_[:, idx1:idx2] = pos_scaler[name].transform(mat[:, idx1:idx2])

    return mat_


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

def get_angles_from_data(data, l_shoulder_idx, r_shoulder_idx):
    l_shoulder_pos = data[:,l_shoulder_idx,[0,2]]
    r_shoulder_pos = data[:,r_shoulder_idx,[0,2]]

    diff = r_shoulder_pos - l_shoulder_pos
    angles = np.arctan2(diff[:,1],diff[:,0]) + ((3 * np.pi)/2)

    return clamp_angles(angles)


def clamp_angles(angles):
    return np.remainder(angles + np.pi, np.pi * 2) - np.pi


def get_angle_between_angles(angles):
    diffs = np.diff(angles)
    greater_than_pi_halfs = 2 * (np.abs(diffs) <= np.pi) - 1
    corrected_diffs = greater_than_pi_halfs * diffs
    corrected_diffs = clamp_angles(corrected_diffs)

    return corrected_diffs

def rotate_multiple_joints(data, angles):
    rotator = R.from_euler("y", angles)

    for curr_joint_idx in range(data.shape[1]):
        data[:, curr_joint_idx, :] = rotator.apply(data[:, curr_joint_idx, :])
    return data

def rotate_single_joint(data, angles):
    rotator = R.from_euler("y", angles[:-data.shape[0]])
    return  rotator.apply(data)

class ParalellMLPProcessor():
    def __init__(self, nr_of_timesteps_per_feature, target_delta_T, augment_rotation_number):
        self.nr_of_timesteps_per_feature = nr_of_timesteps_per_feature
        self.target_delta_t = target_delta_T

        self.inputs = np.array([])
        self.outputs = np.array([])

        self.feet_inputs = np.array([])
        self.feet_outputs = np.array([])

        self.glow_inputs = np.array([])
        self.glow_outputs = np.array([])

        self.heads = np.array([])
        self.min = 1.0
        self.max = 1.0
        self.heading_dirs = None
        self.scaler = {}
        self.scaler_is_not_yet_fitted = True
        self.augment_rotation_number = augment_rotation_number
        self.min_animation_length = 4 #seconds
        self.total_seq_length = int(self.min_animation_length / self.target_delta_t)

    def append_file(self, file_name):

        mocap_importer = MocapImporter.Importer(file_name)
        if not mocap_importer.is_usable(self.min_animation_length + (self.nr_of_timesteps_per_feature * self.target_delta_t)):
            return
        print("imported file :" + file_name)

        global_positions = mocap_importer.zipped_global_positions
        joint_names = mocap_importer.joint_names
        frame_time = mocap_importer.frame_time
        bone_dependencies = mocap_importer.bone_dependencies

        resampled_global_pos = resample_mocap_data(in_delta_t=frame_time, target_delta_t=self.target_delta_t,
                                                   data=global_positions.reshape(global_positions.shape[0], -1))

        resampled_global_pos = resampled_global_pos.reshape(resampled_global_pos.shape[0], -1, 3)

        # resampled_global_pos[:,:,0] = resampled_global_pos[:,:,0] * -1 #evil hack

        #resampled_global_pos = augment_dataset(resampled_global_pos.reshape(-1, 3), self.augment_rotation_number).reshape(-1, resampled_global_pos.shape[1], 3)
        resampled_global_pos = resampled_global_pos
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

        number_of_joints = resampled_global_pos.shape[1]
        heads = resampled_global_pos[:, [head_idx], :]
        resampled_global_pos -= resampled_global_pos[:, [head_idx] * number_of_joints, :]

        heading_directions = get_angles_from_data(resampled_global_pos, l_shoulder_idx, r_shoulder_idx)

        if self.heading_dirs is None:
            self.heading_dirs = heading_directions
        else:
            np.hstack((self.heading_dirs, heading_directions))

        resampled_global_pos = rotate_multiple_joints(resampled_global_pos, heading_directions)

        ff_set = resampled_global_pos[:,
                 [l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx,
                  r_elbow_idx, l_knee_idx, r_knee_idx], :]

        hand_inputs = ff_set[:, [0, 1], :]
        skeletal_outputs = ff_set[:, [2, 3, 4, 5, 6, 7, 8, 9, 10], :]

        ae_set = resampled_global_pos[:, [hip_idx, l_foot_idx, r_foot_idx, l_knee_idx, r_knee_idx], :]

        feet_inputs = ae_set.reshape(ae_set.shape[0], -1)
        feet_outputs = ae_set.reshape(ae_set.shape[0], -1)

        hand_inputs = hand_inputs.reshape(hand_inputs.shape[0], -1)
        skeletal_outputs = skeletal_outputs.reshape(skeletal_outputs.shape[0], -1)
        heads = heads.reshape(heads.shape[0], -1)

        angular_vels = get_angle_between_angles(heading_directions)
        angular_accels = get_angle_between_angles(angular_vels)

        head_vels = rotate_single_joint(np.diff(heads, axis=0), heading_directions)
        head_accels = np.diff(head_vels, axis=0)

        hand_vels = np.diff(hand_inputs, axis=0)
        hand_accels = np.diff(hand_vels, axis=0)

        vels = np.hstack((hand_vels, head_vels, angular_vels.reshape((angular_vels.shape[0], 1))))
        accels = np.hstack((hand_accels, head_accels, angular_accels.reshape((angular_accels.shape[0], 1))))

        rolled_hand_inputs = hand_inputs  # np.hstack((inputs,np.roll(inputs,-1, axis=0)))
        rolled_feet_inputs = feet_inputs
        for i in range(1, self.nr_of_timesteps_per_feature):
            rolled_hand_inputs = np.hstack((rolled_hand_inputs, np.roll(hand_inputs, i * 1, axis=0)))
            rolled_feet_inputs = np.hstack((rolled_feet_inputs, np.roll(feet_inputs, i * 1, axis=0)))

        assert (self.nr_of_timesteps_per_feature >= 2)

        # glow prep

        glow_hand_pos = hand_inputs[2:, :]
        glow_vels = vels[1:, :]
        glow_accels = accels
        glow_outputs = skeletal_outputs[2:,:]
        glow_frames = glow_outputs.shape[0] #since we need accurate accels
        glow_posepoints = skeletal_outputs.shape[1]
        glow_posepoints_offset = glow_posepoints * self.nr_of_timesteps_per_feature
        input_length = glow_hand_pos.shape[1] + glow_vels.shape[1] + glow_accels.shape[1]
        glow_cond_length = glow_posepoints_offset + input_length * (self.nr_of_timesteps_per_feature + 1)

        glow_cond_input = np.concatenate((glow_hand_pos, glow_vels, glow_accels), axis=1)

        rolled_poses = np.empty((glow_frames,  glow_cond_length))
        for i in range(0, self.nr_of_timesteps_per_feature):
            rolled_poses[:, (glow_posepoints * i): (glow_posepoints*(i+1))] = np.roll(glow_outputs, i * -1, axis=0)
            rolled_poses[:, (glow_posepoints_offset + input_length * i):(glow_posepoints_offset + input_length * (i + 1))] = np.roll(glow_cond_input, i * -1, axis=0)

        rolled_poses[:, -input_length:] = np.roll(glow_cond_input, self.nr_of_timesteps_per_feature * -1, axis=0)
        glow_outputs = np.roll(glow_outputs, self.nr_of_timesteps_per_feature * -1, axis=0)

        cutoff = (glow_frames - self.nr_of_timesteps_per_feature) % self.total_seq_length
        glow_inputs = rolled_poses[cutoff:-self.nr_of_timesteps_per_feature, :] #truncate to fulfull consistent size
        glow_outputs = glow_outputs[cutoff:-self.nr_of_timesteps_per_feature, :]
        #end glow prep


        vels = vels[(self.nr_of_timesteps_per_feature + 1):, :]
        accels = accels[self.nr_of_timesteps_per_feature:, :]
        hand_poses = hand_inputs[self.nr_of_timesteps_per_feature + 2:, :]

        rolled_feet_inputs = rolled_feet_inputs[self.nr_of_timesteps_per_feature + 2:-1, :]
        feet_outputs = feet_outputs[self.nr_of_timesteps_per_feature + 3:, :]

        rolled_hand_inputs = rolled_hand_inputs[self.nr_of_timesteps_per_feature+2:, :]
        skeletal_outputs = skeletal_outputs[self.nr_of_timesteps_per_feature+2:, :]
        heads = heads[self.nr_of_timesteps_per_feature+2:, :]

        rolled_hand_inputs = np.hstack((rolled_hand_inputs, vels, accels))
        rolled_feet_inputs = np.hstack((rolled_feet_inputs, hand_poses[:-1, :], vels[:-1, :], accels[:-1, :]))

        if self.inputs.shape[0] == 0:
            self.inputs = rolled_hand_inputs
            self.outputs = skeletal_outputs
            self.heads = heads
            self.feet_inputs = rolled_feet_inputs
            self.feet_outputs = feet_outputs
            self.glow_inputs = glow_inputs
            self.glow_outputs = glow_outputs
        else:
            self.inputs = np.vstack((self.inputs, rolled_hand_inputs))
            self.outputs = np.vstack((self.outputs, skeletal_outputs))
            self.heads = np.vstack((self.heads, heads))
            self.feet_inputs = np.vstack((self.feet_inputs, rolled_feet_inputs))
            self.feet_outputs = np.vstack((self.feet_outputs, feet_outputs))
            self.glow_inputs = np.vstack((self.glow_inputs, glow_inputs))
            self.glow_outputs = np.vstack((self.glow_outputs, glow_outputs))

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

    def __fit_scaler(self):
        if self.scaler_is_not_yet_fitted:
            self.scaler['Head'] = StandardScaler()
            self.scaler['Head'].fit(self.heads)

            self.scaler['LHandVels'] = StandardScaler()
            self.scaler['LHandVels'].fit(self.inputs[:, -20:-17])

            self.scaler['RHandVels'] = StandardScaler()
            self.scaler['RHandVels'].fit(self.inputs[:, -17:-14])

            self.scaler['HeadVels'] = StandardScaler()
            self.scaler['HeadVels'].fit(self.inputs[:, -14:-11])

            self.scaler['HeadingDirVels'] = StandardScaler()
            self.scaler['HeadingDirVels'].fit(self.inputs[:,-11:-10])

            self.scaler['LHandAccels'] = StandardScaler()
            self.scaler['LHandAccels'].fit(self.inputs[:, -10:-7])

            self.scaler['RHandAccels'] = StandardScaler()
            self.scaler['RHandAccels'].fit(self.inputs[:, -7:-4])

            self.scaler['HeadAccels'] = StandardScaler()
            self.scaler['HeadAccels'].fit(self.inputs[:, -4:-1])

            self.scaler['HeadingDirAccels'] = StandardScaler()
            self.scaler['HeadingDirAccels'].fit(self.inputs[:,-1:])

            self.scaler['LeftHand'] = StandardScaler()
            self.scaler['LeftHand'].fit(self.inputs[:, :3])

            self.scaler['RightHand'] = StandardScaler()
            self.scaler['RightHand'].fit(self.inputs[:, 3:6])

            name_list = ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg']
            for idx in range(len(name_list)):
                self.scaler[name_list[idx]] = StandardScaler()
                self.scaler[name_list[idx]].fit(self.outputs[:, idx * 3:(idx + 1) * 3])

            # self.scaler_is_not_yet_fitted = False

    def get_scaled_inputs(self, nr_of_angles=0):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, self.inputs[:, :-20], ['LeftHand', 'RightHand']),
             fit_pos_vector_from_names(self.scaler, self.inputs[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))
        )

    def get_scaled_outputs(self):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, self.outputs, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'])

    def get_scaled_feet_inputs(self, nr_of_angles=0):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, self.feet_inputs[:, :-26], ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg']),
             fit_pos_vector_from_names(self.scaler, self.feet_inputs[:, -26:], ['LeftHand', 'RightHand', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))

        )

    def get_scaled_feet_outputs(self):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, self.feet_outputs, ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'])

    def scale_back_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-20], ['LeftHand', 'RightHand'], inverse=True),
             fit_pos_vector_from_names(self.scaler, data[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels'], inverse=True))
        )

    def scale_back_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=True)

    def scale_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-20], ['LeftHand', 'RightHand']),
             fit_pos_vector_from_names(self.scaler, data[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))
        )
    def scale_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'])

    def scale_feet_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-26], ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg']),
             fit_pos_vector_from_names(self.scaler, data[:, -26:], ['LeftHand', 'RightHand', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))

        )
    def scale_feet_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'])

    def scale_back_feet_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'], inverse=True)

    def scale_inputs_glow(self, data, inverse):
        self.__fit_scaler()
        first_part_end = self.nr_of_timesteps_per_feature * 27
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :first_part_end], ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=inverse),
             fit_pos_vector_from_names(self.scaler, data[:, first_part_end:], ['LeftHand', 'RightHand', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels'], inverse=inverse))
        )

    def scale_outputs_glow(self, data, inverse):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=inverse)

    def get_scaled_inputs_glow(self):
        return self.scale_inputs_glow(self.glow_inputs, False)

    def get_scaled_outputs_glow(self):
        return self.scale_outputs_glow(self.glow_outputs, False)




    def get_scale_back_feet_mats(self):
        h_var = self.scaler['Hips'].scale_
        lf_var = self.scaler['LeftFoot'].scale_
        rf_var = self.scaler['RightFoot'].scale_
        ll_var = self.scaler['LeftLeg'].scale_
        rl_var = self.scaler['RightLeg'].scale_
        vars = np.hstack((h_var,lf_var,rf_var,ll_var,rl_var))

        h_mean = self.scaler['Hips'].mean_
        lf_mean = self.scaler['LeftFoot'].mean_
        rf_mean = self.scaler['RightFoot'].mean_
        ll_mean = self.scaler['LeftLeg'].mean_
        rl_mean = self.scaler['RightLeg'].mean_
        means = np.hstack((h_mean,lf_mean,rf_mean,ll_mean,rl_mean))
        return [vars, means]

    def rotate_back(self, data, start_idx=None, end_idx=None):
        datacopy = np.empty(data.shape)

        if not start_idx is None:
            neg_start_idx = -1 * (self.inputs.shape[0] - start_idx)
            neg_end_idx = -1 * (self.inputs.shape[0] - end_idx)
        else:
            neg_start_idx = -1 * (self.inputs.shape[0])
            neg_end_idx = end_idx

        rotations = self.heading_dirs[neg_start_idx:neg_end_idx]
        rotator = R.from_euler("y", rotations)

        for curr_joint_idx in range(datacopy.shape[1]):
            datacopy[:, curr_joint_idx, :] = rotator.apply(data[:, curr_joint_idx, :], inverse=True)

        return datacopy, rotations

    def add_heads(self, data, start_idx=None, end_idx=None):
        datacopy = data.copy()
        for i in range(data.shape[1]):
            datacopy[:, i, :] += self.heads[start_idx:end_idx]
        return datacopy

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

    def get_global_pos_from_prediction(self, eval_input, target_output, other_preprocessor, start_idx=None, end_idx=None):

        # 'Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg''Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'
        # "l_hand_idx, r_hand_idx, l_elbow_idx, r_elbow_idx, hip_idx, l_foot_idx, r_foot_idx
        #         -         -             0             1             2         3          4            5           6            7            8
        #         0         1             2             3             4         5          6            7           8            9            10
        #                            'LeftArm',     'RightArm',     'Hips' 'LeftFoot' 'RightFoot' 'LeftForeArm' 'RightForeArm' 'LeftLeg','RightLeg'
        # [l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx, r_elbow_idx, l_knee_idx, r_knee_idx]

        bone_dependencies = [[0, 7], [1, 8], [2, 4], [3, 4], [4, -1], [5, 9], [6, 10], [7, 2], [8, 3], [9, 4], [10, 4]]
        bone_dependencies = np.array(bone_dependencies)

        global_positions = np.hstack((other_preprocessor.scale_back_input(eval_input)[:, :6],
                                      other_preprocessor.scale_back_output(target_output)))
        # global_positions = np.hstack((eval_input, eval_output))
        global_positions = global_positions.reshape(global_positions.shape[0], -1, 3)
        global_positions, rotations = self.rotate_back(global_positions, start_idx, end_idx)
        global_positions = self.add_heads(global_positions, start_idx, end_idx)

        return bone_dependencies, global_positions, rotations

    def save(self, target_dir):
        np.savez(target_dir,
                 inputs=self.inputs,
                 outputs=self.outputs,
                 feet_inputs=self.feet_inputs,
                 feet_outputs=self.feet_outputs,
                 nr_of_timesteps_per_feature=self.nr_of_timesteps_per_feature,
                 target_delta_t=self.target_delta_t,
                 heads=self.heads,
                 augment_rotation_number=self.augment_rotation_number,
                 heading_dirs=self.heading_dirs,
                 glow_inputs=self.glow_inputs,
                 glow_outputs=self.glow_outputs)

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
        self.heading_dirs = numpy_importer['heading_dirs']
        self.glow_inputs = numpy_importer['glow_inputs']
        self.glow_outputs = numpy_importer['glow_outputs']