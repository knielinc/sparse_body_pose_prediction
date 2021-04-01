from Helpers import MocapImporter
from Helpers import Animator
import numpy as np
from Helpers.glow import models
from Helpers.glow.generate_config_glow import generate_cfg
from Helpers import SMPLImporter
from scipy.spatial.transform import Rotation as R
from os import listdir
from os.path import isfile, join, isdir, exists
from Helpers import AnimationStacker

def save_anim(reference_positions, bone_dependencies, rotations, name, prep):

    reference_anim = Animator.MocapAnimator2(reference_positions, [''] * 40, bone_dependencies,
                                             prep.target_delta_t,
                                             heading_dirs=rotations,
                                             name=name)
    reference_anim.animation()

STACKCOUNT = 15
TARGET_FPS = 20

# Start Qt event loop unless running in interactive mode.
root_subfolders = eval_files = [13]

# root_subfolders = eval_files = [108,12,20,222,2284,314,319,348,353,518,542,573,582,595,600,610,627,632,637,690,720,721,724,725,731,732,734,735,762,864,870,948,958,971,977]

#655,775,656,2246,823,777,2248,2272,2247,837,
                                 #833,585,961,832,588,964,2298,972,1020,2288,
                                 #232,569,270,309,445,227,1064,886,
                                 # 1455,920,1821,2186,717,723,1453,4,1181,1942,
                                 # 148,250,415,903,1893,2207,478,1354,1373,1381]

for numbr in root_subfolders:
    file = "E:/Master/Sorted Mocap/WALKING/WALKING_" + str(numbr) + ".npz"
    mocap_importer = MocapImporter.Importer(file)

    print("start import file: " + file)
    animation_file = "movies_out/" + file.split('.npz')[0].split('/')[-1] + ".mp4"

    from Helpers import DataPreprocessor
    prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    prep.append_file(file, mirror=True, reverse=False)
    prep.finalize()

    eval_input = prep.scale_input(prep.inputs)  # .scale_input(eval_prep.inputs)
    eval_output = prep.scale_output(prep.outputs)  # scale_output(eval_prep.outputs)

    bone_dependencies, global_positions, rotations = prep.get_global_pos_from_prediction(eval_input, eval_output, prep)
    heads = prep.heads
    head_vels = np.diff(heads, prepend=0, axis=0)

    head_vels_ = prep.inputs[:, -14:-11]
    head_vels = head_vels_

    speeds = np.sqrt(np.square(head_vels[:, 2]) + np.square(head_vels[:, 0]))

    speed_criterion = speeds < 0.025
    dirs_ = np.arctan2(head_vels[:, 2], head_vels[:, 0])
    # head_vels = DataPreprocessor.rotate_single_joint(head_vels, rotations)
    head_vels[speed_criterion] = 1

    dirs = np.arctan2(head_vels[:,2],head_vels[:,0])

    print("processed file")
    print("exporting animation file")

    save_anim(global_positions, bone_dependencies, dirs, animation_file, prep)

    print("finished export to: " +animation_file)



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
