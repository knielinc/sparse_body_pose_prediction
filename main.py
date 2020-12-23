from Helpers import MocapImporter
from Helpers import Animator
import numpy as np

# Start Qt event loop unless running in interactive mode.


mocap_importer = MocapImporter.Importer("C:/Users/cknie/Desktop/Sorted Movement/Basketball/78_01.bvh")

print("feet below arms: " + str(mocap_importer.check_feet_below_arms()))
print("feet below knees: " + str(mocap_importer.check_feet_below_knees()))
mocap_importer.correlation_between_feet_and_hands()

if __name__ == '__main__':
    anim = Animator.MocapAnimator(mocap_importer.zipped_global_positions, mocap_importer.joint_names, mocap_importer.bone_dependencies, mocap_importer.frame_time, write_to_file=False)
    anim.animation()

# anim.save_animation("testSave.mp4")
