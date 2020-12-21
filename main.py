from Helpers import MocapImporter
from Helpers import Animator
import numpy as np

# Start Qt event loop unless running in interactive mode.


mocap_importer = MocapImporter.Importer("C:/Users/cknie/Desktop/convertedMocapData/ACCAD/Female1General_c3d/A14 - stand to skip_poses.npz")

if __name__ == '__main__':
    anim = Animator.MocapAnimator(mocap_importer.zipped_global_positions, mocap_importer.joint_names, mocap_importer.bone_dependencies, mocap_importer.frame_time, write_to_file=False)
    anim.animation()

# anim.save_animation("testSave.mp4")
