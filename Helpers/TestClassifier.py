from Helpers import FileClassifier as fc
from Helpers import MocapImporter

tuples = [("Male2MartialArtsPunches_c3d", "C3 - Run_poses.npz", fc.mocap_dataset.ACCAD, fc.move_type.BOXING),
          ("Male2MartialArtsKicks_c3d","Extended 2_poses.npz",  fc.mocap_dataset.ACCAD, fc.move_type.UNKNOWN),
          ("Male1General_c3d",  "Extended 2_poses.npz",         fc.mocap_dataset.ACCAD, fc.move_type.UNKNOWN),
          ("Male1General_c3d",  "Extended 2_walk.npz",          fc.mocap_dataset.ACCAD, fc.move_type.WALKING),

          ("Subject_3_F_MoSh",  "Subject_19_F_1_poses.npz",     fc.mocap_dataset.BMLMOVI, fc.move_type.WALKING),
          ("Subject_9_F_MoSh",  "Subject_19_F_7_poses.npz",     fc.mocap_dataset.BMLMOVI, fc.move_type.UNKNOWN),
          ("Subject_19_F_MoSh", "Subject_29_F_18_poses.npz",    fc.mocap_dataset.BMLMOVI, fc.move_type.INTERACTION),

          ("rub001",            "0007_normal_jog1_poses.npz",    fc.mocap_dataset.BMLRUB, fc.move_type.UNKNOWN),
          ("rub042",            "0009_knocking1_poses.npz",    fc.mocap_dataset.BMLRUB, fc.move_type.INTERACTION),
          ("rub001",            "0001_treadmill_fast_poses.npz",    fc.mocap_dataset.BMLRUB, fc.move_type.UNKNOWN),
          ("rub001",            "0021_circle_walk_poses.npz",    fc.mocap_dataset.BMLRUB, fc.move_type.WALKING),

          ("05",                "01_04_poses.npz",    fc.mocap_dataset.CMU, fc.move_type.UNKNOWN),
          ("42",                "16_34_poses.npz",    fc.mocap_dataset.CMU, fc.move_type.WALKING),
          ("1337",              "16_43_poses.npz",    fc.mocap_dataset.CMU, fc.move_type.RUNNING),
          ("420",               "26_11_poses.npz",    fc.mocap_dataset.CMU, fc.move_type.INTERACTION),

          ("420",               "HDM_tr_01-01_01_120_poses.npz",    fc.mocap_dataset.MPIHDM05, fc.move_type.WALKING),
          ("420",               "HDM_bk_01-02_01_120_poses.npz",    fc.mocap_dataset.MPIHDM05, fc.move_type.WALKING),
          ("420",               "HDM_bk_03-04_04_120_poses.npz",    fc.mocap_dataset.MPIHDM05, fc.move_type.WORKOUT),
          ("420",               "HDM_tr_02-03_03_120_poses.npz",    fc.mocap_dataset.MPIHDM05, fc.move_type.INTERACTION),
          ("420",               "HDM_tr_01-02_03_120_poses.npz",    fc.mocap_dataset.MPIHDM05, fc.move_type.WALKING),

          ]

idx = 0
correct = 0
for item in tuples:
    folder_name = item[0]
    file_name = item[1]
    dataset = item[2]
    correct_move_type = item[3]

    estimated = fc.classify_file(file_name=file_name, folder_name=folder_name, mocap_dataset_class=dataset)

    symbol = "✓" if estimated == correct_move_type else "✗"
    if estimated == correct_move_type:
        correct += 1
    print("Test " + str(idx)  + ": \t" + symbol + " \t estimated: " + str(estimated) + " correct: " + str(correct_move_type))
    idx += 1

print("\n" + str(correct) + "/" + str(idx) + " correct")


import os
path_tuples = fc.get_path_tuples(src_path="E:\\Master\\Target Mocap")

counter_dict = {}

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
                fc.classify_file(curr_database, subfolder, file)
            mocap_importer = MocapImporter.Importer(sub_path +"\\" + file)
            seconds = mocap_importer.frame_time * mocap_importer.nr_of_frames
            if file_class in counter_dict.keys():
                counter_dict[file_class] += seconds
            else:
                counter_dict[file_class] = seconds

import matplotlib.pyplot as plt
import numpy as np

for key in counter_dict.keys():
    counter_dict[key] = int(counter_dict[key] / 60)

labels = [str(label).split(".")[1] for label in counter_dict.keys() if not label is None]

x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
values = counter_dict.values()

fig, ax = plt.subplots(figsize=(20,10))

rects1 = ax.bar(x, values, width, label='count')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('minutes')
ax.set_title('minutes for each motion type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x(), height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)

fig.tight_layout()

plt.show()