from Helpers import MocapImporter
from Helpers import Animator
import numpy as np

# Start Qt event loop unless running in interactive mode.


# mocap_importer = MocapImporter.Importer("E:/Systemordner/Downloads/Amass Dataset/EyesJapanDataset/Eyes_Japan_Dataset/aita/walk-01-normal-aita_poses.npz")
mocap_importer = np.load("E:/Master/Converted Mocap/TotalCapture/s1/acting1_poses.npz")

global_positions = mocap_importer['zipped_global_positions']
joint_names = mocap_importer['joint_names']
frame_time = mocap_importer['frame_time']
bone_dependencies = mocap_importer['bone_dependencies']



if __name__ == '__main__':
    anim = Animator.MocapAnimator(global_positions, joint_names, bone_dependencies, frame_time)
    anim.animation()

# anim.save_animation("testSave.mp4")
