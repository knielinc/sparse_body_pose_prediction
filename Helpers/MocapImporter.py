from Helpers import BVHImporter
from Helpers import SMPLImporter
import numpy as np
import scipy.stats

class Importer():
    def __init__(self, file_name):
        if(".bvh" in file_name):
            bvh_importer = BVHImporter.BVHImporter(file_name)
            self.joint_names = bvh_importer.joint_names
            self.bone_dependencies = bvh_importer.dependencies
            self.zipped_global_quat_rotations = bvh_importer.zipped_global_quat_rotations
            self.zipped_local_quat_rotations = bvh_importer.zipped_local_quat_rotations
            self.zipped_global_positions = bvh_importer.zipped_global_positions
            self.zipped_local_positions = bvh_importer.zipped_local_positions
            self.frame_time = bvh_importer.frame_time
            self.nr_of_frames = bvh_importer.nr_of_frames
        elif(".npz" in file_name):
            if not ('poses' in np.load(file_name).files or 'zipped_global_positions' in np.load(file_name).files):
                print("tried to load invalid file: " + str(file_name))
                return
            if 'poses' in np.load(file_name).files:
                smpl_importer = SMPLImporter.SMPLImporter(file_name)
                self.joint_names = smpl_importer.joint_names
                self.bone_dependencies = smpl_importer.dependencies
                # self.zipped_global_quat_rotations = smpl_importer.zipped_global_quat_rotations
                # self.zipped_local_quat_rotations = smpl_importer.zipped_local_quat_rotations
                self.zipped_global_positions = smpl_importer.zipped_global_positions
                # self.zipped_local_positions = smpl_importer.zipped_local_positions
                self.frame_time = smpl_importer.frame_time
                self.nr_of_frames = smpl_importer.nr_of_frames
            else:
                numpy_importer = np.load(file_name)

                self.joint_names = list(numpy_importer['joint_names'])
                self.bone_dependencies = numpy_importer['bone_dependencies']
                # self.zipped_global_quat_rotations = smpl_importer.zipped_global_quat_rotations
                # self.zipped_local_quat_rotations = smpl_importer.zipped_local_quat_rotations
                self.zipped_global_positions = numpy_importer['zipped_global_positions']
                # self.zipped_local_positions = smpl_importer.zipped_local_positions
                self.frame_time = numpy_importer['frame_time']
                self.nr_of_frames = numpy_importer['nr_of_frames']
        else:
            pass

    def is_usable(self, min_sec):
        if not (self.check_feet_below_arms() and self.check_feet_below_knees()):
            if not self.check_feet_below_arms():
                print("not usable.. feet below arms")
            if not self.check_feet_below_knees():
                print("not usable.. feet below knees")
            if not self.check_at_least_x_frames(min_sec):
                print("not usable.. not enough frames")


        return self.check_feet_below_arms() and self.check_feet_below_knees() and self.check_at_least_x_frames(min_sec)

    def check_at_least_x_frames(self, min_sec):

        return self.zipped_global_positions.shape[0] > int(min_sec / self.frame_time)

    def check_feet_below_arms(self):
        l_hand_idx   = self.joint_names.index('LeftHand')
        r_hand_idx   = self.joint_names.index('RightHand')
        head_idx     = self.joint_names.index('Head')

        min_hand_pos = np.min(self.zipped_global_positions[:, [l_hand_idx, r_hand_idx, head_idx], 1], axis=1)

        l_foot_idx  = self.joint_names.index('LeftFoot')
        r_foot_idx  = self.joint_names.index('RightFoot')

        max_feet_pos = np.max(self.zipped_global_positions[:, [l_foot_idx, r_foot_idx], 1], axis=1)

        return sum(max_feet_pos > min_hand_pos) == 0

    def check_feet_below_knees(self):
        l_knee_idx = self.joint_names.index('LeftLeg')
        r_knee_idx = self.joint_names.index('RightLeg')

        min_knee_pos = np.min(self.zipped_global_positions[:, [l_knee_idx,r_knee_idx], 1], axis=1)

        l_foot_idx = self.joint_names.index('LeftFoot')
        r_foot_idx = self.joint_names.index('RightFoot')

        max_feet_pos = np.max(self.zipped_global_positions[:, [l_foot_idx, r_foot_idx], 1], axis=1)

        return sum(max_feet_pos > min_knee_pos) == 0

    def correlation_between_feet_and_hands(self):
        m = self.get_centered_arm_and_leg_positions()
        l_arm_poses  = m[:, 0, :].reshape(-1, 3)
        r_arm_poses  = m[:, 1, :].reshape(-1, 3)
        l_foot_poses = m[:, 2, :].reshape(-1, 3)
        r_foot_poses = m[:, 3, :].reshape(-1, 3)

        # llcorr = scipy.stats.spearmanr(l_arm_poses, l_foot_poses)
        # lrcorr = scipy.stats.spearmanr(l_arm_poses, r_foot_poses)
        # rlcorr = scipy.stats.spearmanr(r_arm_poses, l_foot_poses)
        # rrcorr = scipy.stats.spearmanr(r_arm_poses, r_foot_poses)

        llcorr_p = scipy.stats.kendalltau(l_arm_poses, l_foot_poses)
        lrcorr_p = scipy.stats.kendalltau(l_arm_poses, r_foot_poses)
        rlcorr_p = scipy.stats.kendalltau(r_arm_poses, l_foot_poses)
        rrcorr_p = scipy.stats.kendalltau(r_arm_poses, r_foot_poses)

        correlations = np.array([llcorr_p[0], lrcorr_p[0], rlcorr_p[0], rrcorr_p[0]])
        # print(llcorr)
        # print(lrcorr)
        # print(rlcorr)
        # print(rrcorr)

        print(llcorr_p)
        print(lrcorr_p)
        print(rlcorr_p)
        print(rrcorr_p)
        print("mean: " + str(correlations.mean()))

    def get_centered_arm_and_leg_positions(self):
        head_idx    = self.joint_names.index('Head')
        l_hand_idx  = self.joint_names.index('LeftHand')
        r_hand_idx  = self.joint_names.index('RightHand')
        l_foot_idx  = self.joint_names.index('LeftFoot')
        r_foot_idx  = self.joint_names.index('RightFoot')

        return self.zipped_global_positions[:, [l_hand_idx, r_hand_idx, l_foot_idx, r_foot_idx], :] - self.zipped_global_positions[:, [head_idx] * 4, :]


