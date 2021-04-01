from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from Helpers import FileHelpers
from Helpers import DataPreprocessor
from Helpers import StatsPrinter
target_folder = "stats/"
training_prep_folder = "E:/Master/Sorted Mocap/WALKING/"

FileHelpers.create_folder(target_folder)
numbers = [45, 720, 734,  # bwd
           338, 1148, 2112,  # circle
           650, 763, 2308,  # diagonal
           976, 1514, 2016,  # fwd
           12, 13, 772  # sideways
           ]
motiondict = {
    45 : "bwd",
    720 : "bwd",
    734 : "bwd",
    338 : "circle",
    1148 : "circle",
    2112 : "circle",
    650 : "dia",
    763 : "dia",
    2308 : "dia",
    976 : "fwd",
    1514 : "fwd",
    2016 : "fwd",
    12 : "side",
    13 : "side",
    772 : "side",
}


def print_speeds_and_dirs():
    STACKCOUNT = 15
    TARGET_FPS = 20

    default_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    default_prep.append_folder(training_prep_folder, [], mirror=False, reverse=False)
    default_prep.finalize()
    default_name = "default"
    StatsPrinter.print_dirs(default_prep, target_folder, default_name)

    augmented_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    augmented_prep.append_folder(training_prep_folder, [], mirror=True, reverse=True)
    augmented_prep.finalize()
    augmented_name = "augmented"
    StatsPrinter.print_dirs(augmented_prep, target_folder, augmented_name)

    vels = default_prep.inputs[:, -14:-11]
    speeds = np.linalg.norm(vels, axis=1)
    speeds_hist = np.histogram(speeds, 20)

    # print(speeds_hist)

    bins = default_prep.sample_max_bins(default_prep.bins_glow)
    counted_bins = np.bincount(bins + 1)

    print("bins_default: " + str(counted_bins))

    vels = augmented_prep.inputs[:, -14:-11]
    speeds = np.linalg.norm(vels, axis=1)
    speeds_hist = np.histogram(speeds, 20)

    # print(speeds_hist)

    bins = augmented_prep.sample_max_bins(augmented_prep.bins_glow)
    counted_bins = np.bincount(bins + 1)
    print("bins_augmented: " + str(counted_bins))

PATH_PREFIX = training_prep_folder + "WALKING_"
numbers = [45, 720, 734,  # bwd
           338, 1148, 2112,  # circle
           650, 763, 2308,  # diagonal
           976, 1514, 2016,  # fwd
           12, 13, 772  # sideways
           ]

eval_files = [PATH_PREFIX + str(elem) + ".npz" for elem in numbers]

predicted_files_prefix = "E:\\Systemordner\\Dokumente\\Pycharm\Master\\sparse_body_pose_prediction\\predictions_weighted\\saved_animations\\"
predictions = []
methods = ["FF", "GLOW", "RNN2"]
for number in numbers:
    for method in methods:
        filename = predicted_files_prefix + "WALKING_" + str(number) + "_trained_on_WALKING_" + method + "_animation.npz"
        curr_dict = {"filename" : filename,
                     "filenumber": number,
                     "motiontype" : motiondict[number],
                     "method" : method}
        predictions.append(curr_dict)


lower_limbs = ['l_foot', 'r_foot', 'l_elbow', 'r_elbow', 'l_knee', 'r_knee']
upper_limbs = ['l_shoulder', 'r_shoulder', 'hip', 'l_elbow', 'r_elbow']

left_foot = 'l_foot'
right_foot = 'r_foot'

# final_statistics = {"bwd":[],"circle":[],"dia":[],"fwd":[], "side":[]}
motion_types = ["bwd","circle","dia","fwd", "side"]

#loop over motion types
for method in methods:
    print("method: " + method)
    for motion_type in motion_types:
        upper_mse_tot = None
        lower_mse_tot = None
        overall_mse_tot = None
        mean_foot_distance_error_tot = None
        upper_smoothness_tot = None
        lower_smoothness_tot = None
        overall_smoothness_tot = None
        perceptual_loss_tot = None
        perceptual_loss_shuffled_tot = None
        perceptual_loss_mse_rotated_tot = None

        for file in predictions:
            if motion_type != file["motiontype"] or method != file["method"]:
                continue

            filename = file["filename"]
            mat = np.load(filename)

            generated_data = mat['generated_data']
            reference_data = mat['reference_data']
            joint_names = mat['joint_names']
            activity = mat['activity']
            delta_t = mat['delta_t']
            cond_input = mat['cond_input']

            upper_indices = np.where(np.isin(joint_names,upper_limbs))[0]
            lower_indices = np.where(np.isin(joint_names,lower_limbs))[0]
            left_foot_idx = np.where(np.isin(joint_names,left_foot))[0]
            right_foot_idx = np.where(np.isin(joint_names,right_foot))[0]

            def get_mse(difference):
                return np.sum(np.square(difference), axis=2)

            def get_dist(difference):
                return np.linalg.norm((difference), axis=2)
            #mse

            upper_mse = get_mse(generated_data[:,upper_indices] - reference_data[:,upper_indices])
            lower_mse = get_mse(generated_data[:,lower_indices] - reference_data[:,lower_indices])
            overall_mse = get_mse(generated_data - reference_data)

            #foot_distance
            gen_foot_distance = get_dist(generated_data[:,left_foot_idx] - generated_data[:, right_foot_idx])
            ref_foot_distance = get_dist(reference_data[:,left_foot_idx] - reference_data[:, right_foot_idx])
            mean_squared_foot_distance_error = np.square(gen_foot_distance - ref_foot_distance)

            if upper_mse_tot is None:
                upper_mse_tot = upper_mse
                lower_mse_tot = lower_mse
                overall_mse_tot = overall_mse
                mean_foot_distance_error_tot = mean_squared_foot_distance_error
                upper_smoothness_tot = None
                lower_smoothness_tot = None
                overall_smoothness_tot = None
                perceptual_loss_tot = None
                perceptual_loss_shuffled_tot = None
                perceptual_loss_mse_rotated_tot = None
            else:
                np.append(upper_mse_tot, upper_mse, axis=0)
                np.append(lower_mse_tot, lower_mse, axis=0)
                np.append(overall_mse_tot, overall_mse, axis=0)
                np.append(mean_foot_distance_error_tot, mean_squared_foot_distance_error, axis=0)
                upper_smoothness_tot = None
                lower_smoothness_tot = None
                overall_smoothness_tot = None
                perceptual_loss_tot = None
                perceptual_loss_shuffled_tot = None
                perceptual_loss_mse_rotated_tot = None

        upper_mse_tot = upper_mse_tot.mean()
        lower_mse_tot = lower_mse_tot.mean()
        overall_mse_tot = overall_mse_tot.mean()
        mean_foot_distance_error_tot = mean_foot_distance_error_tot.mean()
        print(motion_type)
        print(str(lower_mse_tot) + "\t" + str(upper_mse_tot)  + "\t" + str(overall_mse_tot)  + "\t" + str(mean_foot_distance_error_tot))


