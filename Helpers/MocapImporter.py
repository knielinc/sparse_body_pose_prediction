from Helpers import BVHImporter
from Helpers import SMPLImporter

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
            pass
