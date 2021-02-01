from shutil import copyfile
from os.path import isfile, join, isdir, exists
import os
from Helpers import FileClassifier as fc
from os import listdir, makedirs

src_path="E:\\Master\\Target Mocap"
target_path = "E:\\Master\\Sorted Mocap"
path_tuples = fc.get_path_tuples(src_path=src_path)

counter_dict = {}

counter = 0
for tuple in path_tuples:
    curr_path = tuple[0]
    curr_database = tuple[1]

    subfolders = [f for f in os.listdir(curr_path)if os.path.isdir(curr_path +"\\" + f)]

    for subfolder in subfolders:
        sub_path = curr_path + "\\" + subfolder
        files = [f for f in os.listdir(sub_path) if os.path.isfile(sub_path +"\\" + f)]
        for file in files:
            if file == 'shape.npz':
                continue
            file_class = fc.classify_file(curr_database, subfolder, file)
            if file_class is None:
                continue
            if file_class is fc.move_type.UNKNOWN:
                continue

            if file_class in counter_dict.keys():
                counter_dict[file_class] += 1
            else:
                counter_dict[file_class] = 0

            curr_idx = str(counter_dict[file_class])

            move_type = fc.move_type.type_as_string(file_class)

            src_file_path = sub_path + "\\" + file

            target_file_path = target_path + "\\" + move_type + "\\" + move_type + "_" +  curr_idx + ".npz"

            if not  exists(target_path + "\\" + move_type):
                makedirs(target_path + "\\" + move_type)

            copyfile(src_file_path, target_file_path)
