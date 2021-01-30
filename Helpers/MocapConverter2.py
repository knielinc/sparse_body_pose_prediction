from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
import numpy as np
from Helpers import MocapImporter
from pathlib import Path

target = "E:\Master\Target Mocap"
root_folder = "E:\Master\Source Mocap"
root_subfolders = [f for f in listdir(root_folder) if isdir(join(root_folder, f))]
folders = []
for root_subfolder in root_subfolders:
    folders.append(root_folder + "/" + root_subfolder)

def convert_file(source_dir, target_dir):
    if(".bvh" in source_dir):
        bvh_importer = MocapImporter.Importer(source_dir)
        joint_names = bvh_importer.joint_names
        bone_dependencies = bvh_importer.bone_dependencies
        zipped_global_quat_rotations = bvh_importer.zipped_global_quat_rotations
        zipped_local_quat_rotations = bvh_importer.zipped_local_quat_rotations
        zipped_global_positions = bvh_importer.zipped_global_positions
        zipped_local_positions = bvh_importer.zipped_local_positions
        frame_time = bvh_importer.frame_time
        nr_of_frames = bvh_importer.nr_of_frames
        np.savez(target_dir[:-4], joint_names=joint_names, bone_dependencies=bone_dependencies, zipped_local_positions=zipped_local_positions, frame_time=frame_time, nr_of_frames=nr_of_frames, zipped_global_positions=zipped_global_positions, zipped_global_quat_rotations=zipped_global_quat_rotations, zipped_local_quat_rotations=zipped_local_quat_rotations)

    elif(".npz" in source_dir):
        smpl_importer = MocapImporter.Importer(source_dir)
        joint_names = smpl_importer.joint_names
        bone_dependencies = smpl_importer.bone_dependencies
        # self.zipped_global_quat_rotations = smpl_importer.zipped_global_quat_rotations
        # self.zipped_local_quat_rotations = smpl_importer.zipped_local_quat_rotations
        zipped_global_positions = smpl_importer.zipped_global_positions
        # self.zipped_local_positions = smpl_importer.zipped_local_positions
        frame_time = smpl_importer.frame_time
        nr_of_frames = smpl_importer.nr_of_frames
        np.savez(target_dir, joint_names=joint_names, bone_dependencies=bone_dependencies, zipped_global_positions=zipped_global_positions, frame_time=frame_time, nr_of_frames=nr_of_frames)


for root_subfolder in root_subfolders:
    path = root_folder + "/" + root_subfolder
    allfiles = [f for f in listdir(path) if isfile(join(path, f))]
    subfolders =  [f for f in listdir(path) if isdir(join(path, f))]
    print("🗀" + path)

    for file in allfiles:
        sourcedir = path + "/" + file
        targetdir = target + "/" + root_subfolder + "/" + file
        Path(target + "/" + root_subfolder).mkdir(parents=True, exist_ok=True)
        if (exists(targetdir)):
            print("\t\t\t Skipped file : " + file)
            continue
        # if not 'poses' in np.load(sourcedir).files:
        #     print("\t\t\t Skipped invalid file : " + file)
        #     continue
        convert_file(sourcedir, targetdir)
        print("\t\t\t Converted file : " + file)

    for subfolder in subfolders:

        subpath = path+"/"+subfolder
        allfiles = [f for f in listdir(subpath) if isfile(join(subpath, f))]

        print("\t\t └🗀" + subfolder)

        for file in allfiles:
            sourcedir = subpath + "/" + file
            targetdir = target + "/" + root_subfolder + "/" + subfolder + "/" + file
            if(exists(targetdir)):
                print("\t\t\t Skipped file : " + file)
                continue

            if not 'poses' in np.load(sourcedir).files:
                print("\t\t\t Skipped invalid file : " + file)
                continue
            if (not exists(target + "/" + root_subfolder + "/" + subfolder)):
                makedirs(target + "/" + root_subfolder + "/" + subfolder)
            convert_file(sourcedir, targetdir)
            print("\t\t\t Converted file : " + file)


