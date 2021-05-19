from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from Helpers import FileHelpers, ModelWrappers
from Helpers import DataPreprocessor
from Helpers import StatsPrinter
target_folder = "stats/"
training_prep_folder = "E:/Master/Sorted Mocap/WALKING/"
from sklearn.utils import shuffle

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

predicted_files_prefix = "E:\\Systemordner\\Dokumente\\Pycharm\Master\\sparse_body_pose_prediction\\moglow_dropout\\stats\\"
predictions = []
methods = ["FF", "RNN2", "GLOW"]
for number in numbers:
    for method in methods:
        filename = predicted_files_prefix + "WALKING_" + str(number) + "_trained_on_WALKING_" + method + "_animation.npz"
        curr_dict = {"filename" : filename,
                     "filenumber": number,
                     "motiontype" : motiondict[number],
                     "method" : method}
        predictions.append(curr_dict)


lower_limbs = ['l_foot', 'r_foot', 'l_knee', 'r_knee']
upper_limbs = ['l_shoulder', 'r_shoulder', 'hip', 'l_elbow', 'r_elbow']
feet = ['l_shoulder', 'r_shoulder', 'hip', 'l_elbow', 'r_elbow']
knees = ['l_knee', 'r_knee']

left_foot = 'l_foot'
right_foot = 'r_foot'

# final_statistics = {"bwd":[],"circle":[],"dia":[],"fwd":[], "side":[]}
motion_types = ["fwd","bwd","side","dia","circle"]

# prep = DataPreprocessor.ParalellMLPProcessor(15, 1.0 / 20, 5)
# prep.append_folder(training_prep_folder, [], mirror=True, reverse=True)
# prep.finalize()
#
# gan_wrapper = ModelWrappers.gan_wrapper(prep)
# gan_wrapper.train(5, 4000, 0.0001)

#loop over motion types
lines = []
for method in methods:
    print("method: " + method)


    for motion_type in motion_types:
        upper_mse_tot = None
        feet_mse_tot = None
        knee_mse_tot = None
        overall_mse_tot = None
        mean_foot_distance_error_tot = None
        upper_smoothness_tot = None
        feet_smoothness_tot = None
        knee_smoothness_tot = None
        overall_smoothness_tot = None
        perceptual_loss_tot = None
        perceptual_loss_shuffled_tot = None
        perceptual_loss_mse_rotated_tot = None
        bone_length_tot = None

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
            feet_indices = np.where(np.isin(joint_names,feet))[0]
            knee_indices = np.where(np.isin(joint_names,knees))[0]
            lower_indices = np.where(np.isin(joint_names,lower_limbs))[0]
            left_foot_idx = np.where(np.isin(joint_names,left_foot))[0]
            right_foot_idx = np.where(np.isin(joint_names,right_foot))[0]

            def get_mse(difference):
                return np.sum(np.square(difference), axis=2)

            def get_dist(difference):
                return np.linalg.norm((difference), axis=2)
            #mse

            upper_mse = get_mse(generated_data[:,upper_indices] - reference_data[:,upper_indices])
            feet_mse = get_mse(generated_data[:,feet_indices] - reference_data[:,feet_indices])
            knee_mse = get_mse(generated_data[:,knee_indices] - reference_data[:,knee_indices])
            overall_mse = get_mse(generated_data - reference_data)

            #foot_distance
            gen_foot_distance = get_dist(generated_data[:,left_foot_idx] - generated_data[:, right_foot_idx])
            ref_foot_distance = get_dist(reference_data[:,left_foot_idx] - reference_data[:, right_foot_idx])
            mean_squared_foot_distance_error = np.abs(gen_foot_distance - ref_foot_distance)

            #smoothness
            frame_length = generated_data.shape[0]
            ones = np.ones(frame_length - 1)
            diag_elems = np.ones(frame_length) * -2
            diag_elems[0] += 1
            diag_elems[-1] += 1
            smoothness_mat = np.diag(ones, 1) + np.diag(ones, -1) + np.diag(diag_elems)

            def get_smoothness(smoothness_mat, joint_matrix):
                smoothness = None
                for joint in range(joint_matrix.shape[2]):
                    # print(np.linalg.norm(np.matmul(smoothness_mat, joint_matrix[:, :, joint]), axis=1))
                    if smoothness is None:
                        smoothness = np.linalg.norm(np.matmul(smoothness_mat, joint_matrix[:, :, joint]), axis=1)
                        # print(smoothness)
                    else:
                        smoothness += np.linalg.norm(np.matmul(smoothness_mat, joint_matrix[:, :, joint]), axis=1)
                        # print(smoothness)
                smoothness /= joint_matrix.shape[2]
                return smoothness[1:-1]



            def get_feet_length(feet_pos):

                original_feet_pos = feet_pos #* scaling_vars + scaling_means


                left_foot_idx = np.where(np.isin(joint_names, left_foot))[0].item()
                right_foot_idx = np.where(np.isin(joint_names, right_foot))[0].item()

                left_knee_idx = np.where(np.isin(joint_names, 'l_knee'))[0].item()
                right_knee_idx = np.where(np.isin(joint_names, 'r_knee'))[0].item()

                hip_idx = np.where(np.isin(joint_names, 'hip'))[0].item()

                left_elbow_idx = np.where(np.isin(joint_names, 'l_elbow'))[0].item()
                right_elbow_idx = np.where(np.isin(joint_names, 'r_elbow'))[0].item()

                left_shoulder_idx = np.where(np.isin(joint_names, 'l_shoulder'))[0].item()
                right_shoulder_idx = np.where(np.isin(joint_names, 'r_shoulder'))[0].item()

                # 'Hips', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'
                # hip - LeftLeg
                h_ll = original_feet_pos[:, hip_idx] - original_feet_pos[:, left_knee_idx]
                # hip - RightLeg
                h_rl = original_feet_pos[:, hip_idx] - original_feet_pos[:, right_knee_idx]

                # LeftLeg - LeftFoot
                ll_lf = original_feet_pos[:, left_knee_idx] - original_feet_pos[:, left_foot_idx]
                # RightLeg - RightFoot
                rl_rf = original_feet_pos[:, right_knee_idx] - original_feet_pos[:, right_foot_idx]


                # RightLeg - RightFoot
                rs_re = original_feet_pos[:, right_shoulder_idx] - original_feet_pos[:, right_elbow_idx]

                # RightLeg - RightFoot
                ls_le = original_feet_pos[:, left_shoulder_idx] - original_feet_pos[:, left_elbow_idx]

                get_length = lambda x: np.sum(x * x, 1)

                h_ll = get_length(h_ll)

                h_rl = get_length(h_rl)
                ll_lf = get_length(ll_lf)
                rl_rf = get_length(rl_rf)

                rs_re = get_length(rs_re)
                ls_le = get_length(ls_le)

                return (np.hstack((h_ll, h_rl, ll_lf, rl_rf, rs_re, ls_le)))

            generated_feet_length = get_feet_length(generated_data[:, :])
            default_feet_length = get_feet_length(reference_data[:, :])

            bone_length_differences = np.abs(generated_feet_length - default_feet_length)#.mean()

            upper_smoothness = get_smoothness(smoothness_mat, generated_data[:,upper_indices])
            lower_smoothness = get_smoothness(smoothness_mat, generated_data[:,feet_indices])
            overall_smoothness = get_smoothness(smoothness_mat, generated_data[:,:])

            shuffled_reference, shuffled_generated = shuffle(reference_data, generated_data)

            def get_rotated_data(gen, ref):
                diff_vec = ref - gen
                mse = np.linalg.norm(diff_vec, axis=2)

                random_vectors = (np.random.rand(gen.shape[0], gen.shape[1],
                                                gen.shape[2]) * 2) - 1
                rand_mse = np.linalg.norm(random_vectors, axis=2)

                factor = mse / rand_mse
                random_vectors *= factor[:, :, np.newaxis]

                return ref - random_vectors

            rotated_gen = get_rotated_data(generated_data, reference_data)

            # def get_perceptual_loss(generated_data, reference_data, perceptual_loss_net, cond_input):
            #     cond_input_ = prep.scale_cond(cond_input)
            #     generated_data_ = prep.scale_output(generated_data)
            #     reference_data_ = prep.scale_output(reference_data)
            #     return perceptual_loss_net.get_loss(cond_input_, generated_data_, reference_data_)
            #
            # def get_gan_loss(generated_data, perceptual_loss_net, cond_input):
            #     cond_input_ = prep.scale_cond(cond_input)
            #     generated_data_ = prep.scale_output(generated_data)
            #     return perceptual_loss_net.get_other_loss(cond_input_, generated_data_)
            #
            # perceptual_loss = np.array([get_perceptual_loss(generated_data, reference_data, gan_wrapper, cond_input)])
            # perceptual_loss_shuffled = np.array([get_perceptual_loss(shuffled_generated, shuffled_reference, gan_wrapper, cond_input)])
            # perceptual_loss_mse_rotated = np.array([get_perceptual_loss(rotated_gen, reference_data, gan_wrapper, cond_input)])

            if upper_mse_tot is None:
                upper_mse_tot = upper_mse
                feet_mse_tot = feet_mse
                knee_mse_tot = knee_mse
                overall_mse_tot = overall_mse
                mean_foot_distance_error_tot = mean_squared_foot_distance_error
                upper_smoothness_tot = upper_smoothness
                lower_smoothness_tot = lower_smoothness
                overall_smoothness_tot = overall_smoothness
                bone_length_tot = bone_length_differences
                # perceptual_loss_tot = perceptual_loss
                # perceptual_loss_shuffled_tot = perceptual_loss_shuffled
                # perceptual_loss_mse_rotated_tot = perceptual_loss_mse_rotated
            else:
                upper_mse_tot = np.append(upper_mse_tot, upper_mse, axis=0)
                feet_mse_tot = np.append(feet_mse_tot, feet_mse, axis=0)
                knee_mse_tot = np.append(knee_mse_tot, knee_mse, axis=0)
                overall_mse_tot = np.append(overall_mse_tot, overall_mse, axis=0)
                mean_foot_distance_error_tot = np.append(mean_foot_distance_error_tot, mean_squared_foot_distance_error, axis=0)
                upper_smoothness_tot = np.append(upper_smoothness_tot, upper_smoothness, axis=0)
                lower_smoothness_tot = np.append(lower_smoothness_tot, lower_smoothness, axis=0)
                overall_smoothness_tot = np.append(overall_smoothness_tot, overall_smoothness, axis=0)
                bone_length_tot = np.append(bone_length_tot, bone_length_differences, axis=0)
                # np.append(perceptual_loss_tot, perceptual_loss, axis=0)
                # np.append(perceptual_loss_shuffled_tot, perceptual_loss_shuffled, axis=0)
                # np.append(perceptual_loss_mse_rotated_tot, perceptual_loss_mse_rotated, axis=0)

        upper_mse_tot = upper_mse_tot.mean()
        feet_mse_tot = feet_mse_tot.mean()
        knee_mse_tot = knee_mse_tot.mean()
        overall_mse_tot = overall_mse_tot.mean()
        mean_foot_distance_error_tot = mean_foot_distance_error_tot.mean()
        upper_smoothness_tot = upper_smoothness_tot.mean()
        lower_smoothness_tot = lower_smoothness_tot.mean()
        overall_smoothness_tot = overall_smoothness_tot.mean()
        bone_length_tot = bone_length_differences.mean()

        print(motion_type)
        debug_string = str(feet_mse_tot) + "\t" + str(knee_mse_tot) + "\t" + str(upper_mse_tot)  + "\t" + str(overall_mse_tot)  + "\t" + str(mean_foot_distance_error_tot) + "\t" + str(lower_smoothness_tot) + "\t" + str(bone_length_tot)
        print(debug_string)#+ "\t" + str(perceptual_loss_tot) + "\t" + str(perceptual_loss_shuffled_tot)+ "\t" + str(perceptual_loss_mse_rotated_tot))
        lines.append(debug_string)
print()
out2 = []
for i in range(5):
    for j in range(methods.__len__()):
        out2.append(lines[j * 5 + i])
    empty_string = " \t \t \t \t \t \t "
    out2.append(empty_string)
    out2.append(empty_string)


for line in out2:
    print(line)

for number in numbers:
    fig, ax = plt.subplots(figsize=(15, 5))
    ground_truth_was_printed = False
    for method in methods:
        for file in predictions:
            if number != file["filenumber"] or method != file["method"]:
                continue
            else:
                filename = file["filename"]
                mat = np.load(filename)

                generated_data = mat['generated_data']
                reference_data = mat['reference_data']
                joint_names = mat['joint_names']
                activity = mat['activity']
                delta_t = mat['delta_t']
                cond_input = mat['cond_input']


                t = np.arange(0.0, generated_data.shape[0], 1)* delta_t

                ll = np.where(np.isin(joint_names, 'l_foot'))[0].item()
                rl = np.where(np.isin(joint_names, 'r_foot'))[0].item()

                # rl = joint_names.index('l_foot')
                # ll = joint_names.index('r_foot')

                sample_feet_dists = np.linalg.norm(generated_data[:, ll, :] - generated_data[:, rl, :], axis=1)
                reference_feet_dists = np.linalg.norm(reference_data[:, ll, :] - reference_data[:, rl, :], axis=1)
                # fig, ax = plt.subplots(figsize=(10, 3))

                mydict = {
                    "RNN2":"Recurrent network",
                    "FF":"Feedforward network",
                    "GLOW":"MoGlow network"
                }
                mydict_2 = {
                    "RNN2":0,
                    "FF":1,
                    "GLOW":2
                }
                colors =  np.array([[200,0,0],[100,200,100],[0,100,255],[0,0,0]]) / 255

                if not ground_truth_was_printed:
                    ax.plot(t, reference_feet_dists, color=colors[-1], label='Ground truth', linewidth=2,zorder=10)
                    ground_truth_was_printed = True
                ax.plot(t, sample_feet_dists, color=colors[mydict_2[method]], label=mydict[method], linewidth=1.5)





    ax.legend()

    ax.set(xlabel='time (s)', ylabel='distance (m)',
           title='Distance between Feet over Time')
    ax.grid()
    fig.savefig("stats/" + str(number) + "_stats.pdf")