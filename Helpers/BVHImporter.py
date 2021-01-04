import sys
import bvh
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

flatten = lambda t: [item for sublist in t for item in sublist]


class BVHImporter:
    def __init__(self, file_name):
        FRAMES, FRAME_TIME, joint_names, dependencies, zipped_global_quat_rotations, zipped_local_quat_rotations, zipped_global_positions, zipped_local_positions = load_file(
            file_name)
        self.joint_names = joint_names
        self.dependencies = dependencies
        self.zipped_global_quat_rotations = zipped_global_quat_rotations
        self.zipped_local_quat_rotations = zipped_local_quat_rotations
        self.zipped_global_positions = zipped_global_positions
        self.zipped_local_positions = zipped_local_positions
        self.frame_time = FRAME_TIME
        self.nr_of_frames = FRAMES


def load_file(file_name):
    with open(file_name) as f:
        mocap = bvh.Bvh(f.read())

    FRAMES = mocap.nframes
    FRAME_TIME = mocap.frame_time

    joint_names = mocap.get_joints_names()

    JOINT_COUNT = joint_names.__len__()

    joint_channels = []
    joint_offsets = []
    joint_channels = []
    frame_joint_channels = []
    frame_joint_quaternions = []
    frame_positions = []

    for joint_name in joint_names:
        joint_channels.append(mocap.joint_channels(joint_name))
        joint_offsets.append(mocap.joint_offset(joint_name))

    # joint_id parent_id name offset
    dependency = []
    for joint_name in joint_names:
        currRow = []
        joint_idx = joint_names.index(joint_name)
        currRow.append(joint_idx)
        if (joint_idx == 0):
            currRow.append(-1)
        else:
            currRow.append(joint_names.index(mocap.get_joint(joint_name).parent.name))
        # currRow.append(joint_name)
        for coord in joint_offsets[joint_idx]:
            currRow.append(coord)
        dependency.append(currRow)

    # rotations local_positions global_positions(wrt root)

    frames = np.array(mocap.frames).astype(float)
    root_positions = frames[:, 0:3]
    euler_rotations = frames[:, 3:]

    dependencies = np.array(dependency)

    offsets = dependencies[:, -3:]
    zipped_euler_rotations = []
    zipped_local_rotations = []
    zipped_local_positions = []
    zipped_global_positions = []
    zipped_global_rotations = []
    zipped_local_quat_rotations = []
    zipped_global_quat_rotations = []

    for curr_joint in range(JOINT_COUNT):
        start_idx = 3 * curr_joint
        end_idx = 3 * (curr_joint + 1)
        curr_joint_euler_rots = euler_rotations[:, start_idx:end_idx]
        flipped_euler_rotations = np.flip(curr_joint_euler_rots, 1)
        curr_local_rotations = R.from_euler('xyz', flipped_euler_rotations, degrees=True)
        curr_joint_offset = np.array(offsets[curr_joint])
        curr_local_positions = curr_local_rotations.apply(curr_joint_offset)

        if (curr_joint == 0):
            curr_global_rotations = curr_local_rotations
            curr_global_positions = root_positions
        else:
            curr_parent_rotation = zipped_global_rotations[dependencies[curr_joint, -4].astype(int)];
            curr_global_rotations = curr_parent_rotation * curr_local_rotations
            curr_global_positions = curr_parent_rotation.apply(curr_joint_offset) + zipped_global_positions[
                dependencies[curr_joint, -4].astype(int)]

        zipped_euler_rotations.append(curr_joint_euler_rots)
        zipped_local_rotations.append(curr_local_rotations)
        zipped_local_positions.append(curr_local_positions)
        zipped_global_positions.append(curr_global_positions)
        zipped_global_rotations.append(curr_global_rotations)
        zipped_local_quat_rotations.append(curr_local_rotations.as_quat())
        zipped_global_quat_rotations.append(curr_global_rotations.as_quat())

    #print("lul")
    zipped_global_positions = np.moveaxis(np.array(zipped_global_positions), 0, 1)
    zipped_local_positions = np.moveaxis(np.array(zipped_local_positions), 0, 1)
    zipped_global_quat_rotations = np.moveaxis(np.array(zipped_global_quat_rotations), 0, 1)
    zipped_local_quat_rotations = np.moveaxis(np.array(zipped_local_quat_rotations), 0, 1)
    return FRAMES, FRAME_TIME, joint_names, dependencies[:, :2].astype('int'), zipped_global_quat_rotations, zipped_local_quat_rotations, zipped_global_positions, zipped_local_positions
