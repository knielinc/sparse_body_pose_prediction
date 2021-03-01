from Helpers import MocapImporter
from Helpers import Animator
import numpy as np
from Helpers.glow import models
from Helpers.glow.generate_config_glow import generate_cfg
from Helpers import SMPLImporter
from scipy.spatial.transform import Rotation as R
from os import listdir
from os.path import isfile, join, isdir, exists

# Start Qt event loop unless running in interactive mode.
root_subfolders = eval_files = [ "E:/Master/Sorted Mocap/BASKETBALL/BASKETBALL_10.npz",
               "E:/Master/Sorted Mocap/BOXING/BOXING_64.npz",
               "E:/Master/Sorted Mocap/WALKING/WALKING_2265.npz",
               "E:/Master/Sorted Mocap/THROWING/THROWING_58.npz",
               "E:/Master/Sorted Mocap/INTERACTION/INTERACTION_1534.npz"
               ]

for file in root_subfolders:
    mocap_importer = MocapImporter.Importer(file)

    anim = Animator.MocapAnimator2(mocap_importer.zipped_global_positions, mocap_importer.joint_names,
                                   mocap_importer.bone_dependencies, mocap_importer.frame_time, write_to_file=False, name=file.split('.npz')[0].split('/')[-1] + ".mp4")
    anim.animation()



# print("feet below arms: " + str(mocap_importer.check_feet_below_arms()))
# print("feet below knees: " + str(mocap_importer.check_feet_below_knees()))
# mocap_importer.correlation_between_feet_and_hands()
'''
rots = mocap_importer.rots
default_pos = mocap_importer.default_pos
dependencies = mocap_importer.dependencies_
trans_allframes = mocap_importer.trans_allframes

mocap_importer.zipped_global_positions[:] = default_pos[:22]

zipped_euler_rotations = []
zipped_local_rotations = []
zipped_local_positions = []
zipped_global_positions = []
zipped_global_rotations = []
zipped_local_quat_rotations = []
zipped_global_quat_rotations = []


for curr_joint in range(rots.shape[1]):

    curr_joint_euler_rots = rots[:,curr_joint]

    # flipped_euler_rotations = np.flip(curr_joint_euler_rots, 1)
    curr_local_rotations = R.from_matrix(curr_joint_euler_rots)
    curr_joint_offset = np.array(default_pos[curr_joint])
    curr_local_positions = curr_local_rotations.apply(curr_joint_offset)

    if (curr_joint == 0):
        curr_global_rotations = curr_local_rotations
        curr_global_positions = trans_allframes
    else:
        curr_parent_rotation = zipped_global_rotations[dependencies[curr_joint].astype(int)]
        curr_global_rotations = curr_parent_rotation * curr_local_rotations
        curr_global_positions = curr_parent_rotation.apply(curr_joint_offset) + zipped_global_positions[
            dependencies[curr_joint].astype(int)]

    zipped_euler_rotations.append(curr_joint_euler_rots)
    zipped_local_rotations.append(curr_local_rotations)
    zipped_local_positions.append(curr_local_positions)
    zipped_global_positions.append(curr_global_positions)
    zipped_global_rotations.append(curr_global_rotations)
    zipped_local_quat_rotations.append(curr_local_rotations.as_quat())
    zipped_global_quat_rotations.append(curr_global_rotations.as_quat())

zipped_global_positions = np.moveaxis(np.array(zipped_global_positions), 0, 1)
zipped_local_positions = np.moveaxis(np.array(zipped_local_positions), 0, 1)
zipped_global_quat_rotations = np.moveaxis(np.array(zipped_global_quat_rotations), 0, 1)
zipped_local_quat_rotations = np.moveaxis(np.array(zipped_local_quat_rotations), 0, 1)
'''




# anim.save_animation("testSave.mp4")
