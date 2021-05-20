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
    angles = np.arctan2(diff[:,1],diff[:,0]) + ((np.pi)/2)

    return clamp_angles(angles)


def clamp_angles(angles):
    return np.remainder(angles + np.pi, TWO_PI) - np.pi

def get_angle_between_angles(angles):
    diffs = np.diff(angles)
    greater_than_pi_halfs = 2 * (np.abs(diffs) <= np.pi) - 1
    corrected_diffs = greater_than_pi_halfs * diffs
    corrected_diffs = clamp_angles(corrected_diffs)

    return corrected_diffs

def rotate_multiple_joints(data, angles):
    rotator = R.from_euler("y", angles)

    for curr_joint_idx in range(data.shape[1]):
        data[:, curr_joint_idx, :] = rotator.apply(data[:, curr_joint_idx, :], inverse=False)
    return data

def rotate_single_joint(data, angles):
    rotator = R.from_euler("y", angles[:data.shape[0]])
    return  rotator.apply(data, inverse=False)

class ParalellMLPProcessor():
    def __init__(self, nr_of_timesteps_per_feature, target_delta_T, augment_rotation_number, use_weighted_sampling=True):
        self.use_weighted_sampling = use_weighted_sampling
        self.nr_of_timesteps_per_feature = nr_of_timesteps_per_feature
        self.target_delta_t = target_delta_T

        self.inputs = None
        self.outputs = None

        self.feet_inputs = None
        self.feet_outputs = None

        self.glow_inputs = None
        self.glow_outputs = None

        self.heads = None
        self.min = 1.0
        self.max = 1.0
        self.heading_dirs = None
        self.scaler = {}
        self.scaler_is_not_yet_fitted = True
        self.augment_rotation_number = augment_rotation_number
        self.min_animation_length = 2 #seconds
        self.total_seq_length = int(self.min_animation_length / self.target_delta_t)

        self.bins_glow = None
        self.bins_default = None
        self.usable_clips = [ [] for _ in range(33) ]

    def append_file(self, file_name,  mirror=False, reverse=False):
        mocap_importer = MocapImporter.Importer(file_name)
        if not mocap_importer.is_usable(self.min_animation_length + (self.nr_of_timesteps_per_feature * self.target_delta_t)):
            return
        print("imported file :" + file_name)

        global_positions = mocap_importer.zipped_global_positions
        joint_names = mocap_importer.joint_names
        frame_time = mocap_importer.frame_time
        bone_dependencies = mocap_importer.bone_dependencies

        global_positions[:, :, 0] = global_positions[:, :, 0] * -1  # evil hack


        resampled_global_pos = resample_mocap_data(in_delta_t=frame_time, target_delta_t=self.target_delta_t,
                                                   data=global_positions.reshape(global_positions.shape[0], -1))

        resampled_global_pos = resampled_global_pos.reshape(resampled_global_pos.shape[0], -1, 3)
        # if not mirror:
        self.append_motion(joint_names, frame_time, bone_dependencies, resampled_global_pos, file_name, False)

        # data augmentation
        import copy
        #reverse motion
        if reverse:
            reversed_resampled_global_pos = np.flip(resampled_global_pos, axis=0)
            self.append_motion(joint_names, frame_time, bone_dependencies, reversed_resampled_global_pos)

        #flip motion
        if mirror:
            flipped_resampled_global_pos = copy.deepcopy(resampled_global_pos)
            flipped_resampled_global_pos[:,:,2] = flipped_resampled_global_pos[:,:,2] * -1
            joint_names_flipped = []
            for name_idx in range(joint_names.__len__()):
                currname = joint_names[name_idx]
                if "Left" in currname:
                    joint_names_flipped.append(currname.replace("Left", "Right"))
                elif "Right" in currname:
                    joint_names_flipped.append(currname.replace("Right", "Left"))
                else:
                    joint_names_flipped.append(currname)
            self.append_motion(joint_names_flipped, frame_time, bone_dependencies, flipped_resampled_global_pos)

        #flip and reverse
        if reverse and mirror:
            reversed_flipped_resampled_global_pos = np.flip(flipped_resampled_global_pos, axis=0)
            self.append_motion(joint_names_flipped, frame_time, bone_dependencies, reversed_flipped_resampled_global_pos)

    def get_bins(self, dir):
        directions = dir[:,[0, 2]]

        speeds = np.sqrt(np.square(directions[:, 1]) + np.square(directions[:, 0]))
        speed_criterion = speeds > 0.01
        directions_ = directions[speed_criterion]

        N = 32

        angles = np.arctan2(directions_[:, 1], directions_[:, 0])
        angles = angles + 4 * np.pi - (2 * np.pi / (N * 2))
        angles = np.remainder(angles, 2 * np.pi)

        ranges = np.linspace(0, 2 * np.pi + 2 * np.pi / N, N + 1, endpoint=False)
        # dir_hist = np.histogram(angles, ranges)[0]
        dir_bins = np.digitize(angles, ranges)

        all_bins = np.ones(speeds.shape[0]) * -1
        all_bins[speed_criterion] = dir_bins - 1

        return all_bins.astype(int)

    def append_motion(self, joint_names, frame_time, bone_dependencies, resampled_global_pos_in, file_name="", is_modified=True):

        resampled_global_pos = resampled_global_pos_in.copy()
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
        heads_global_pos = resampled_global_pos[:, [head_idx], :]
        resampled_global_pos -= resampled_global_pos[:, [head_idx] * number_of_joints, :]

        heading_directions = get_angles_from_data(resampled_global_pos, l_shoulder_idx, r_shoulder_idx)

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
        heads_global_pos = heads_global_pos.reshape(heads_global_pos.shape[0], -1)


        angular_vels = get_angle_between_angles(heading_directions)
        angular_accels = get_angle_between_angles(angular_vels)

        global_vels = np.diff(heads_global_pos, axis=0)
        head_vels = rotate_single_joint(global_vels, heading_directions)
        head_accels = np.diff(head_vels, axis=0)

        hist_buckets = self.get_bins(head_vels)

        hand_vels = np.diff(hand_inputs, axis=0)
        hand_accels = np.diff(hand_vels, axis=0)

        head_heights = heads_global_pos[:, 1:2]
        vels = np.hstack((hand_vels, head_vels, angular_vels.reshape((angular_vels.shape[0], 1))))
        accels = np.hstack((hand_accels, head_accels, angular_accels.reshape((angular_accels.shape[0], 1))))

        rolled_hand_inputs = np.empty((hand_inputs.shape[0],  hand_inputs.shape[1] * self.nr_of_timesteps_per_feature))
        rolled_feet_inputs = np.empty((feet_inputs.shape[0],  feet_inputs.shape[1] * self.nr_of_timesteps_per_feature))
        for i in range(0, self.nr_of_timesteps_per_feature):
            rolled_hand_inputs[:, (hand_inputs.shape[1] * i): (hand_inputs.shape[1]*(i+1))] = np.roll(hand_inputs, i * 1, axis=0)
            rolled_feet_inputs[:, (feet_inputs.shape[1] * i): (feet_inputs.shape[1]*(i+1))] = np.roll(feet_inputs, i * 1, axis=0)

        # assert (self.nr_of_timesteps_per_feature >= 2)

        # glow prep
        glow_hand_pos = hand_inputs[2:, :]
        glow_head_height = head_heights[2:,:]
        glow_vels = vels[1:, :]
        glow_bins = hist_buckets[1:]
        glow_accels = accels
        glow_outputs = skeletal_outputs[2:,:]
        glow_frames = glow_outputs.shape[0] #since we need accurate accels
        glow_posepoints = skeletal_outputs.shape[1]
        glow_posepoints_offset = glow_posepoints * self.nr_of_timesteps_per_feature
        input_length = glow_hand_pos.shape[1] + glow_vels.shape[1] + glow_accels.shape[1] + glow_head_height.shape[1]
        glow_cond_length = glow_posepoints_offset + input_length * (self.nr_of_timesteps_per_feature + 1)

        glow_cond_input = np.concatenate((glow_hand_pos, glow_head_height, glow_vels, glow_accels), axis=1)

        rolled_poses = np.empty((glow_frames,  glow_cond_length))
        for i in range(0, self.nr_of_timesteps_per_feature):
            rolled_poses[:, (glow_posepoints * i): (glow_posepoints*(i+1))] = np.roll(glow_outputs, i * -1, axis=0)
            rolled_poses[:, (glow_posepoints_offset + input_length * i):(glow_posepoints_offset + input_length * (i + 1))] = np.roll(glow_cond_input, i * -1, axis=0)

        rolled_poses[:, -input_length:] = np.roll(glow_cond_input, self.nr_of_timesteps_per_feature * -1, axis=0)
        glow_outputs = np.roll(glow_outputs, self.nr_of_timesteps_per_feature * -1, axis=0)

        glow_cutoff = (glow_frames - self.nr_of_timesteps_per_feature) % self.total_seq_length
        glow_inputs = rolled_poses[glow_cutoff:-self.nr_of_timesteps_per_feature, :] #truncate to fulfull consistent size
        glow_outputs = glow_outputs[glow_cutoff:-self.nr_of_timesteps_per_feature, :]

        glow_bins = glow_bins[glow_cutoff:-self.nr_of_timesteps_per_feature]

        if glow_bins.shape[0] > 80 and glow_bins.shape[0] < 400 and is_modified == False:
            current_bin = np.bincount(glow_bins + 1).argmax()
            number = file_name.split('_')[-1].split('.npz')[0]
            unique, counts = np.unique(glow_bins + 1, return_counts=True)
            mydict = dict(zip(unique, counts))
            percentage = mydict[current_bin] / glow_bins.shape[0] * 100
            still_frames = glow_bins[glow_bins == -1].shape[0]
            total_frames = glow_bins.shape[0]

            if current_bin != 0:
                if(still_frames == total_frames):
                    percentage = 0
                else:
                    percentage *= total_frames / (total_frames - still_frames)

            string_percentage = "{percentage:.1f}".format(percentage=percentage)
            self.usable_clips[current_bin].append(string_percentage + "% for " + number)
        #end glow prep

        def do_cutoff(features):
            cutoff = (features.shape[0] % self.total_seq_length)
            return features[cutoff:]

        def to_mixed_list(numpy_array):
            return [elem for elem in numpy_array]

        vels = vels[(self.nr_of_timesteps_per_feature + 1):, :]
        default_bins = hist_buckets[(self.nr_of_timesteps_per_feature + 1):]
        accels = accels[self.nr_of_timesteps_per_feature:, :]
        hand_poses = hand_inputs[self.nr_of_timesteps_per_feature + 2:, :]
        head_heights_hand = head_heights[self.nr_of_timesteps_per_feature + 2:, :]

        head_heights_feet = head_heights[self.nr_of_timesteps_per_feature + 2:-1, :]
        rolled_feet_inputs = rolled_feet_inputs[self.nr_of_timesteps_per_feature + 2:-1, :]
        feet_outputs = do_cutoff(feet_outputs[self.nr_of_timesteps_per_feature + 3:, :])

        rolled_hand_inputs = rolled_hand_inputs[self.nr_of_timesteps_per_feature + 2:, :]
        skeletal_outputs = do_cutoff(skeletal_outputs[self.nr_of_timesteps_per_feature + 2:, :])
        heads_global_pos = do_cutoff(heads_global_pos[self.nr_of_timesteps_per_feature + 2:, :])

        rolled_hand_inputs = do_cutoff(np.hstack((rolled_hand_inputs, head_heights_hand, vels, accels)))
        rolled_feet_inputs = do_cutoff(np.hstack((rolled_feet_inputs, head_heights_feet, hand_poses[:-1, :], vels[:-1, :], accels[:-1, :])))

        heading_directions = do_cutoff(heading_directions[self.nr_of_timesteps_per_feature + 2:])
        default_bins = do_cutoff(default_bins)

        if self.inputs is None:
            self.inputs       = to_mixed_list(rolled_hand_inputs)
            self.outputs      = to_mixed_list(skeletal_outputs)
            self.heads        = to_mixed_list(heads_global_pos)
            self.feet_inputs  = to_mixed_list(rolled_feet_inputs)
            self.feet_outputs = to_mixed_list(feet_outputs)
            self.glow_inputs  = to_mixed_list(glow_inputs)
            self.glow_outputs = to_mixed_list(glow_outputs)
            self.heading_dirs = to_mixed_list(heading_directions)
            self.bins_glow    = to_mixed_list(glow_bins)
            self.bins_default = to_mixed_list(default_bins)

        else:
            self.inputs.extend(to_mixed_list(rolled_hand_inputs))
            self.outputs.extend(to_mixed_list(skeletal_outputs))
            self.heads.extend(to_mixed_list(heads_global_pos))
            self.feet_inputs.extend(to_mixed_list(rolled_feet_inputs))
            self.feet_outputs.extend(to_mixed_list(feet_outputs))
            self.glow_inputs.extend(to_mixed_list(glow_inputs))
            self.glow_outputs.extend(to_mixed_list(glow_outputs))
            self.heading_dirs.extend(to_mixed_list(heading_directions))
            self.bins_glow.extend(to_mixed_list(glow_bins))
            self.bins_default.extend(to_mixed_list(default_bins))


    def finalize(self):

        # for i in range(self.usable_clips.__len__()):
        #
        #     print("bin: " + str(i))
        #     print(self.usable_clips[i])
        #     print("")

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        self.feet_inputs = np.array(self.feet_inputs)
        self.feet_outputs = np.array(self.feet_outputs)

        self.glow_inputs = np.array(self.glow_inputs)
        self.glow_outputs = np.array(self.glow_outputs)

        self.heads = np.array(self.heads)
        self.heading_dirs = np.array(self.heading_dirs)

        self.bins_glow = np.array(self.bins_glow)
        self.bins_default = np.array(self.bins_default)

        self.create_odds_for_sampling()

    def create_odds_for_sampling(self):
        max_bins_glow = self.sample_max_bins(self.bins_glow)
        self.glow_odds_cumsum = self.create_cumsum_odds_from_bins(max_bins_glow)

        max_bins_default = self.sample_max_bins(self.bins_default)
        self.default_odds_cumsum = self.create_cumsum_odds_from_bins(max_bins_default)

    def sample_max_bins(self, bins):
        reshaped_glow = bins.reshape(-1, self.total_seq_length) + 1
        def get_max(arr):
            return np.bincount(arr).argmax() - 1

        max_bins = np.apply_along_axis(get_max, 1, reshaped_glow)
        return max_bins

    def create_cumsum_odds_from_bins(self, bins):
        counted_bins = np.bincount(bins + 1)
        total_count = bins.shape[0]
        probabilities = np.zeros(total_count)
        for i in range(counted_bins.shape[0]):
            if counted_bins[i] == 0:
                probabilities[bins + 1 == i] = 0
            else:
                probabilities[bins + 1 == i] = counted_bins.max() / counted_bins[i]
        cum_sum = np.cumsum(probabilities)
        return cum_sum

    def sample_weighted_clip(self, sample_count, cum_sums):
        samples = []
        for i in range(sample_count):
            samples.append(self.sample_cum_sum_array(cum_sums))
        return samples

    def sample_clip(self, sample_count, cum_sums):
        return np.random.choice(cum_sums.shape[0], sample_count)

    def sample_glow(self, sample_count):
        if self.use_weighted_sampling:
            return self.sample_weighted_clip(sample_count, self.glow_odds_cumsum)
        else:
            return self.sample_clip(sample_count, self.glow_odds_cumsum)

    def sample_default(self, sample_count):
        count = int(sample_count / self.total_seq_length)
        clip_indices = None
        if self.use_weighted_sampling:
            clip_indices = self.sample_weighted_clip(count, self.default_odds_cumsum)
        else:
            clip_indices = self.sample_clip(count, self.default_odds_cumsum)

        output = list([item for clip_index in clip_indices for item in range(clip_index, clip_index + self.total_seq_length)])
        return output


    def sample_cum_sum_array(self, cumsum):
        idx = np.random.random(1) * np.max(cumsum)
        return np.searchsorted(cumsum, idx).item()

    def append_folder(self, folder, ignore_files=[], mirror=False, reverse=False):
        folders = glob(folder)

        for path in folders:
            allfiles = [f for f in listdir(path) if isfile(join(path, f))]

            for file in allfiles:
                f = path + file
                if not f in ignore_files:
                    self.append_file(f, mirror=mirror, reverse=reverse)
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

            self.scaler['HeadHeight'] = StandardScaler()
            self.scaler['HeadHeight'].fit(self.inputs[:,6:7])

            name_list = ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg']
            for idx in range(len(name_list)):
                self.scaler[name_list[idx]] = StandardScaler()
                self.scaler[name_list[idx]].fit(self.outputs[:, idx * 3:(idx + 1) * 3])

            # self.scaler_is_not_yet_fitted = False

    def scale_cond(self, data):
        # inputs[:, :6], inputs[:, -20:]
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data,['LeftHand', 'RightHand', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels'])


    def get_scaled_inputs(self, nr_of_angles=0):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, self.inputs[:, :-21], ['LeftHand', 'RightHand']),
             fit_pos_vector_from_names(self.scaler, self.inputs[:,-21:-20], ['HeadHeight']),
             fit_pos_vector_from_names(self.scaler, self.inputs[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))
        )

    def get_scaled_outputs(self):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, self.outputs, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'])

    def get_scaled_feet_inputs(self, nr_of_angles=0):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, self.feet_inputs[:, :-27], ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg']),
             fit_pos_vector_from_names(self.scaler, self.feet_inputs[:, -27:-26], ['HeadHeight']),
             fit_pos_vector_from_names(self.scaler, self.feet_inputs[:, -26:], ['LeftHand', 'RightHand', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))

        )

    def get_scaled_feet_outputs(self):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, self.feet_outputs, ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'])

    def scale_back_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-21], ['LeftHand', 'RightHand'], inverse=True),
             fit_pos_vector_from_names(self.scaler, data[:, -21:-20], ['HeadHeight'], inverse=True),
             fit_pos_vector_from_names(self.scaler, data[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels'], inverse=True))
        )

    def scale_back_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=True)

    def scale_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-21], ['LeftHand', 'RightHand']),
             fit_pos_vector_from_names(self.scaler, data[:, -21:-20], ['HeadHeight']),
             fit_pos_vector_from_names(self.scaler, data[:, -20:], ['LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels']))
        )
    def scale_output(self, data):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'])

    def scale_feet_input(self, data):
        self.__fit_scaler()
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :-27], ['Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg']),
             fit_pos_vector_from_names(self.scaler, data[:, -27:-26], ['HeadHeight']),
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
             fit_pos_vector_from_names(self.scaler, data[:, first_part_end:], ['LeftHand', 'RightHand', 'HeadHeight', 'LHandVels', 'RHandVels', 'HeadVels', 'HeadingDirVels', 'LHandAccels', 'RHandAccels', 'HeadAccels', 'HeadingDirAccels'], inverse=inverse))
        )

    def scale_outputs_glow(self, data, inverse):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=inverse)

    def get_scaled_inputs_glow(self):
        return self.scale_inputs_glow(self.glow_inputs, False)

    def get_scaled_outputs_glow(self):
        return self.scale_outputs_glow(self.glow_outputs, False)


    def scale_inputs_glow_test(self, data, inverse):
        self.__fit_scaler()
        first_part_end = self.nr_of_timesteps_per_feature * 27
        return np.hstack(
            (fit_pos_vector_from_names(self.scaler, data[:, :first_part_end], ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=inverse),
             fit_pos_vector_from_names(self.scaler, data[:, first_part_end:], ['HeadVels', 'HeadingDirVels', 'HeadAccels', 'HeadingDirAccels'], inverse=inverse))
        )

    def scale_outputs_glow_test(self, data, inverse):
        self.__fit_scaler()
        return fit_pos_vector_from_names(self.scaler, data, ['LeftArm', 'RightArm', 'Hips', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg'], inverse=inverse)


    def get_glow_test_inputs(self):
        first_part_end = self.nr_of_timesteps_per_feature * 27
        offset_cond = 26

        glow_test_inputs = self.glow_inputs
        for idx in range(self.nr_of_timesteps_per_feature - 1, -1, -1):
            elem_idx = offset_cond * idx
            lhand_idx = elem_idx
            rhand_idx = elem_idx + 1
            lhandvel_idx = elem_idx + 2
            rhandvel_idx = elem_idx + 3
            #head + 4
            #heading + 5
            lhand_accel_idx = elem_idx + 6
            rhand_accel_idx = elem_idx + 7
            glow_test_inputs = np.delete(glow_test_inputs, [lhand_idx, rhand_idx, lhandvel_idx, rhandvel_idx, lhand_accel_idx, rhand_accel_idx], 1)

        return glow_test_inputs

    def get_scaled_inputs_glow_test(self):
        glow_test_inputs = self.get_glow_test_inputs()

        return self.scale_inputs_glow_test(glow_test_inputs, False)

    def get_scaled_outputs_glow_test(self):
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
            if neg_end_idx == 0:
                neg_end_idx = None
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

    def get_global_pos_from_prediction(self, eval_input, target_output, other_preprocessor, start_idx=None, end_idx=None, scale=True):

        # 'Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg''Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'
        # "l_hand_idx, r_hand_idx, l_elbow_idx, r_elbow_idx, hip_idx, l_foot_idx, r_foot_idx
        #         -         -             0             1             2         3          4            5           6            7            8
        #         0         1             2             3             4         5          6            7           8            9            10
        #                            'LeftArm',     'RightArm',     'Hips' 'LeftFoot' 'RightFoot' 'LeftForeArm' 'RightForeArm' 'LeftLeg','RightLeg'
        # [l_hand_idx, r_hand_idx, l_shoulder_idx, r_shoulder_idx, hip_idx, l_foot_idx, r_foot_idx, l_elbow_idx, r_elbow_idx, l_knee_idx, r_knee_idx]

        bone_dependencies = [[0, 7], [1, 8], [2, 4], [3, 4], [4, -1], [5, 9], [6, 10], [7, 2], [8, 3], [9, 4], [10, 4]]
        bone_dependencies = np.array(bone_dependencies)

        if scale:
            global_positions = np.hstack((other_preprocessor.scale_back_input(eval_input)[:, :6],
                                          other_preprocessor.scale_back_output(target_output)))
        else:
            global_positions = np.hstack((eval_input[:, :6],
                                          target_output))
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
                 glow_outputs=self.glow_outputs,
                 bins_glow = self.bins_glow,
                 bins_default = self.bins_default
        )

    def export_unity(self, target_dir):
        motion = []
        #pos
        motion.append(self.heads)
        #rot
        motion.append(self.heading_dirs)

        #lhand
        lhand = self.inputs[:, :3]
        motion.append(lhand)
        #lhanddir
        lelbow = self.outputs[:, 15:18]
        lhandDir = lhand - lelbow
        motion.append(lhandDir)

        #rhand
        rhand = self.inputs[:, 3:6]
        motion.append(rhand)
        #rhanddir
        relbow = self.outputs[:, 18:21]
        rhandDir = rhand - relbow
        motion.append(rhandDir)

        with open(target_dir, 'wb') as f:
            for row in motion:
                np.savetxt(f, [row.reshape(-1)], delimiter=',')

    def load_np(self, source_dir):
        numpy_importer = np.load(source_dir)

        self.inputs = numpy_importer['inputs']
        self.outputs = numpy_importer['outputs']
        self.feet_inputs = numpy_importer['feet_inputs']
        self.feet_outputs = numpy_importer['feet_outputs']
        self.nr_of_timesteps_per_feature = numpy_importer['nr_of_timesteps_per_feature'].item()
        self.target_delta_t = numpy_importer['target_delta_t'].item()
        self.heads = numpy_importer['heads']
        self.augment_rotation_number = numpy_importer['augment_rotation_number'].item()
        self.heading_dirs = numpy_importer['heading_dirs']
        self.glow_inputs = numpy_importer['glow_inputs']
        self.glow_outputs = numpy_importer['glow_outputs']
        self.bins_glow = numpy_importer['bins_glow']
        self.bins_default = numpy_importer['bins_default']

        self.create_odds_for_sampling()

    def save_scalers(self, target_dir):
        from pickle import dump
        self.__fit_scaler()
        dump(self.scaler, open(target_dir + '/scalers.pkl', 'wb'))

    def load_scalers(self, source_dir):
        from pickle import load
        self.scaler_is_not_yet_fitted = False
        self.scaler = load(open(source_dir + "/" + 'scalers.pkl', 'rb'))
