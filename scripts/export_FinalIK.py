from Helpers import DataPreprocessor
from Helpers import ModelWrappers
from Helpers import Models
import numpy as np

STACKCOUNT = 15
TARGET_FPS = 20

'''
    eval_files = [ "E:/Master/Sorted Mocap/WALKING/WALKING_42.npz",
                   "E:/Master/Sorted Mocap/WALKING/WALKING_360.npz",
                   "E:/Master/Sorted Mocap/WALKING/WALKING_420.npz",
                   "E:/Master/Sorted Mocap/WALKING/WALKING_1337.npz",
                   "E:/Master/Sorted Mocap/WALKING/WALKING_2265.npz",
'''

numbers = [45, 720, 734,  # bwd
           338, 1148, 2112,  # circle
           650, 763, 2308,  # diagonal
           976, 1514, 2016,  # fwd
           12, 13, 772  # sideways
           ]
folder = "E:\\Systemordner\\Dokumente\\Pycharm\\Master\\sparse_body_pose_prediction\\moglow_dropout\\unity_motion_export\\"

for filenr in numbers:
    data_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    data_prep.append_file("E:/Master/Sorted Mocap/WALKING/" + "WALKING_" + str(filenr) + ".npz")

    data_prep.finalize()
    data_prep.export_unity(folder + "WALKING_" + str(filenr) + "_trained_on_WALKING_IK" + ".ume")

'''
def exportLine(f):
    floatvals = [float(i) for i in f.readline().split(',')]
    return np.array(floatvals).reshape(-1,3)

with open("E:\\Systemordner\\Dokumente\\Pycharm\\Master\\sparse_body_pose_prediction\\"  + "WALKING_" + str(filenr) + "_unity_out"+ ".csv", 'r') as f:
    lShoulder_f = exportLine(f)
    rShoulder_f = exportLine(f)
    lElbow_f = exportLine(f)
    rElbow_f = exportLine(f)
    hips_f = exportLine(f)
    lKnee_f = exportLine(f)
    rKnee_f = exportLine(f)
    lFoot_f = exportLine(f)
    rFoot_f = exportLine(f)

finalIK_output = np.hstack((lShoulder_f, rShoulder_f, hips_f, lFoot_f, rFoot_f, lElbow_f, rElbow_f, lKnee_f, rKnee_f))

bone_dependencies, global_positions, rotations = data_prep.get_global_pos_from_prediction(data_prep.inputs, finalIK_output,
                                                                             data_prep, scale=False)
_, reference_positions, rotations = data_prep.get_global_pos_from_prediction(data_prep.inputs, data_prep.outputs,
                                                                             data_prep, scale=False)

name = "finalIK" + file_name

from Helpers import Animator
from Helpers import AnimationStacker

anim = Animator.MocapAnimator2(global_positions, [''] * 40, bone_dependencies, data_prep.target_delta_t,
                               heading_dirs=rotations,
                               name="trained.mp4")
anim.animation()
reference_anim = Animator.MocapAnimator2(reference_positions, [''] * 40, bone_dependencies,
                                         data_prep.target_delta_t,
                                         heading_dirs=rotations,
                                         name="reference.mp4")
reference_anim.animation()
AnimationStacker.concatenate_animations("trained.mp4", "reference.mp4", name + ".mp4")
'''
