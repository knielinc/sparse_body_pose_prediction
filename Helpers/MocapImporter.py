from Helpers import BVHImporter
from Helpers import SMPLImporter
import numpy as np

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

                self.joint_names = numpy_importer['joint_names']
                self.bone_dependencies = numpy_importer['bone_dependencies']
                # self.zipped_global_quat_rotations = smpl_importer.zipped_global_quat_rotations
                # self.zipped_local_quat_rotations = smpl_importer.zipped_local_quat_rotations
                self.zipped_global_positions = numpy_importer['zipped_global_positions']
                # self.zipped_local_positions = smpl_importer.zipped_local_positions
                self.frame_time = numpy_importer['frame_time']
                self.nr_of_frames = numpy_importer['nr_of_frames']
        else:
            pass
