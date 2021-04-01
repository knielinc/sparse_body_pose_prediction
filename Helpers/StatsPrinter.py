import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Helpers import FileHelpers as fh
TAB = "\t"

def save_feet_distance(generated_data, reference_data, joint_names, activity, ax):
    t = np.arange(0.0, generated_data.shape[0], 1)

    rl = joint_names.index('l_foot')
    ll = joint_names.index('r_foot')

    sample_feet_dists       = np.linalg.norm(generated_data[:,ll,:] - generated_data[:,rl,:], axis=1)
    reference_feet_dists    = np.linalg.norm(reference_data[:,ll,:] - reference_data[:,rl,:], axis=1)
    # fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, sample_feet_dists, color='green', label='estimated')
    ax.plot(t, reference_feet_dists, color='red', label='ground_truth')

    ax.legend()

    ax.set(xlabel='time (s)', ylabel='distance (m)',
           title='Distance between Feet over Time')
    ax.grid()

def save_feet_velocity(generated_data, reference_data, joint_names, activity, ax, delta_t):
    t = np.arange(0.0, generated_data.shape[0], 1)

    rl = joint_names.index('l_foot')
    ll = joint_names.index('r_foot')

    titles = ['feet velocity',
              'feet acceleration',
              'feet jerk']
    orders = [1,2,3]

    for order in orders:
        if(order == 0):
            generated_data_ = generated_data
            reference_data_ = reference_data
        else:
            generated_data_ = np.diff(generated_data, n=int(order), axis=0) / pow(delta_t, order)
            reference_data_ = np.diff(reference_data, n=int(order), axis=0) / pow(delta_t, order)

        ax_ = ax[order - 1]

        sample_feet_dists       = (np.linalg.norm(generated_data_[:,ll,:], axis=1) + np.linalg.norm(generated_data_[:,rl,:], axis=1)) / 2
        reference_feet_dists    = (np.linalg.norm(reference_data_[:,ll,:], axis=1) + np.linalg.norm(reference_data_[:,rl,:], axis=1)) / 2

        t = np.arange(0.0, generated_data_.shape[0], 1)

        ax_.plot(t, sample_feet_dists, color='#264653', label='estimated')
        ax_.plot(t, reference_feet_dists, color='#E76F51', label='ground_truth')

        ax_.legend()

        y_label = 'm/s'
        if order > 1:
            y_label += str(order)

        ax_.set(xlabel='time (s)', ylabel=y_label,
               title=titles[order - 1])
        ax_.grid()


def save_mse_joint_distance_txt(generated_data, reference_data, joint_names, activity):
    maxheight = np.min(reference_data[:,:,2]) * -1.7
    scaling = 1 # / maxheight
    norms = np.linalg.norm(scaling * (generated_data - reference_data), axis=2)

    file = open(activity + ".txt", "w")
    # \n is placed to indicate EOL (End of Line)

    file.write(activity + "\n")

    for joint_idx in range(generated_data.shape[1]):
        file.writelines(TAB + joint_names[joint_idx]+ "\n")
        file.writelines(TAB + TAB + "max dist: "    + str(np.max(norms[:,[joint_idx]]))+ "\n")
        file.writelines(TAB + TAB + "min dist: "    + str(np.min(norms[:,[joint_idx]]))+ "\n")
        file.writelines(TAB + TAB + "mean dist: "   + str(np.mean(norms[:,[joint_idx]]))+ "\n")
        file.writelines(TAB + TAB + "median dist: " + str(np.median(norms[:,[joint_idx]]))+ "\n")

    file.close()  # to change file access modes


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def save_mse_joint_derivative(generated_data, reference_data, joint_names, activity, order, ax, delta_t):
    maxheight = np.min(reference_data[:,:,2]) * -1.7
    scaling = 1 # / maxheight

    if(order == 0):
        generated_data_ = generated_data
        reference_data_ = reference_data
    else:
        generated_data_ = np.diff(generated_data, n=int(order), axis=0)# / pow(delta_t, order)
        reference_data_ = np.diff(reference_data, n=int(order), axis=0)# / pow(delta_t, order)

    norms = np.linalg.norm(scaling * (generated_data_ - reference_data_), axis=2)

    width = 0.2
    padding = 0.1
    offset = (order - 2) * (width + padding)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']

    plot = ax.boxplot(norms, positions = np.array(range(len(joint_names)))*2 + offset, widths=width, showfliers=False, labels=[''] * joint_names.__len__())
    ax.grid()
    set_box_color(plot, colors[order])

# Some fake data to plot
A= [[1, 2, 5,],  [7, 2]]
B = [[5, 7, 2, 2, 5], [7, 2, 5]]
C = [[3,2,5,7], [6, 7, 3]]

def save_mse_joint_distance(generated_data, reference_data, joint_names, activity, ax, delta_t):
    save_mse_joint_derivative(generated_data, reference_data, joint_names, activity, 0, ax, delta_t)

def save_mse_joint_velocity(generated_data, reference_data, joint_names, activity, ax, delta_t):
    save_mse_joint_derivative(generated_data, reference_data, joint_names, activity, 1, ax, delta_t)

def save_mse_joint_accelleration(generated_data, reference_data, joint_names, activity, ax, delta_t):
    save_mse_joint_derivative(generated_data, reference_data, joint_names, activity, 2, ax, delta_t)

def save_mse_joint_jerk(generated_data, reference_data, joint_names, activity, ax, delta_t):
    save_mse_joint_derivative(generated_data, reference_data, joint_names, activity, 3, ax, delta_t)

def save_mse_joint_text(generated_data, reference_data, activity,  file_name, order, delta_t):
    maxheight = np.min(reference_data[:,:,2]) * -1.7
    scaling = 1 # / maxheight

    if(order == 0):
        generated_data_ = generated_data
        reference_data_ = reference_data
    else:
        generated_data_ = np.diff(generated_data, n=int(order), axis=0) / pow(delta_t, order)
        reference_data_ = np.diff(reference_data, n=int(order), axis=0) / pow(delta_t, order)

    norms = np.linalg.norm((generated_data_ - reference_data_), axis=2)

    lower = np.mean(norms[:,[3,4,7,8]])
    upper = np.mean(norms[:,[0,1,2,5,6]])
    file_lower = file_name + "_order_" + str(order) + "_lower.txt"
    loss_line = activity + " loss mse: " + str(lower)
    fh.append_line(file_lower, loss_line)

    file_upper = file_name + "_order_" + str(order) + "_upper.txt"
    loss_line = activity + " loss mse: " + str(upper)
    fh.append_line(file_upper, loss_line)

def save_perceptual_loss(generated_data, reference_data, activity, perceptual_loss_file_name, perceptual_loss_net, cond_input=None):

    perceptual_loss_file = open(perceptual_loss_file_name, 'a')
    if cond_input is None:
        loss_line = activity + " loss: " + str(
            perceptual_loss_net.get_loss(generated_data[:, 2:, :], reference_data[:, 2:, :])) + "\n"
    else:
        loss_line = activity + " loss: " + str(
            perceptual_loss_net.get_loss(cond_input, generated_data, reference_data)) + "\n"
    perceptual_loss_file.write(loss_line)
    # print(loss_line)
    perceptual_loss_file.close()

def save_perceptual_loss_gan(generated_data, reference_data, activity, perceptual_loss_file_name, perceptual_loss_net):

    perceptual_loss_file = open(perceptual_loss_file_name, 'a')
    loss_line = activity + " loss: " + str(
        perceptual_loss_net.get_loss(generated_data[:, 2:, :], reference_data[:, 2:, :])) + "\n"
    perceptual_loss_file.write(loss_line)
    # print(loss_line)
    perceptual_loss_file.close()

def save_feet_distance_text(generated_data, reference_data, joint_names, activity, file_name):
    rl = joint_names.index('l_foot')
    ll = joint_names.index('r_foot')

    sample_feet_dists       = np.linalg.norm(generated_data[:,ll,:] - generated_data[:,rl,:], axis=1)
    reference_feet_dists    = np.linalg.norm(reference_data[:,ll,:] - reference_data[:,rl,:], axis=1)
    # fig, ax = plt.subplots(figsize=(10, 3))
    norms = np.mean(np.abs(sample_feet_dists - reference_feet_dists))

    lower = np.mean(norms)

    loss_line = activity + " loss mean distance: " + str(lower)
    fh.append_line(file_name, loss_line)

default_print_stats_path = "predictions_non_weighted"

def print_stats(generated_data, reference_data, joint_names, activity, delta_t, perceptual_loss_net, save_matrices=True, target_path=default_print_stats_path, cond_input=None):
    activity = target_path + "/" + activity.split('/')[-1]
    if save_matrices:
        np.savez(activity+"_animation.npz",
                 generated_data=generated_data,
                 reference_data=reference_data,
                 joint_names=joint_names,
                 activity=activity,
                 delta_t=delta_t,
                 cond_input=cond_input
                 )
        fh.append_line(target_path + "/animation_list.txt", activity+"_animation.npz")

    save_mse_joint_distance_txt(generated_data, reference_data, joint_names, activity)

    fig, ax = plt.subplots(5, figsize=(15, 30))

    ax[0].set_title('mse joint distance(m)')
    ax[0].set_xticklabels(joint_names[2:])
    ax[0].set_xticks(np.arange(joint_names.__len__()) * 2)

    save_mse_joint_distance(generated_data[:,2:], reference_data[:,2:], joint_names[2:], activity, ax[0], delta_t)
    save_mse_joint_velocity(generated_data[:,2:], reference_data[:,2:], joint_names[2:], activity, ax[0], delta_t)
    save_mse_joint_accelleration(generated_data[:,2:], reference_data[:,2:], joint_names[2:], activity, ax[0], delta_t)
    save_mse_joint_jerk(generated_data[:,2:], reference_data[:,2:], joint_names[2:], activity, ax[0], delta_t)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
    from matplotlib.patches import Patch
    legend_names = ['position (m)', 'velocity (m/s)', 'acceleration (m/s^2)', 'jerk (m/s^3)']

    legend_elements = [Patch(color=colors[0], label=legend_names[0]),
                       Patch(color=colors[1], label=legend_names[1]),
                       Patch(color=colors[2], label=legend_names[2]),
                       Patch(color=colors[3], label=legend_names[3])]

    ax[0].legend(handles=legend_elements)

    save_feet_distance(generated_data, reference_data, joint_names, activity, ax[1])
    save_feet_velocity(generated_data, reference_data, joint_names, activity, [ax[2], ax[3], ax[4]], delta_t)

    fig.savefig(activity + "_stats.pdf")

    save_mse_joint_text(generated_data, reference_data, activity,  target_path + "/mse", 0, delta_t)
    save_mse_joint_text(generated_data, reference_data, activity,  target_path + "/mse", 1, delta_t)
    save_mse_joint_text(generated_data, reference_data, activity,  target_path + "/mse", 2, delta_t)
    save_mse_joint_text(generated_data, reference_data, activity,  target_path + "/mse", 3, delta_t)

    perceptual_loss_file_name = target_path + "/perceptual_loss.txt"
    if not perceptual_loss_net is None:
        save_perceptual_loss(generated_data, reference_data, activity, perceptual_loss_file_name, perceptual_loss_net, cond_input)

    save_feet_distance_text(generated_data, reference_data, joint_names, activity, target_path + "/feet_distances.txt")

def create_summary(src_folder, target_folder=None, name="overview"):
    if target_folder is None:
        target_folder = src_folder

    def create_plot(file_name, axes, index, y_lim=None):
        ax = axes#[index]
        file = open(file_name, 'r')

        models = []
        loss_Vals = []

        for line in file:
            strings = line.split(' ')
            model = strings[0].split('/')[-1].split('_')
            model = model[0] + '_' + model[-1]
            loss_val = strings[-1]
            loss_val_val = float(loss_val)
            models.append(model.split('_')[0])
            loss_Vals.append(loss_val_val)

        file.close()
        # fig, ax = plt.subplots(figsize=(50,20))
        # ax.yaxis.set_major_formatter(formatter)
        # ax.xticks(fontsize=7)
        title = file_name.split('/')[-1].split('.')[0]
        labels = ['GLOW', 'FF', 'VAE', 'RNN', 'RNN2']
        types = ['Basketball', 'Boxing', 'Walking', 'Throwing', 'Interaction']
        x = np.arange(len(labels))
        width = 0.18
        ax.set_xticklabels(labels)
        ax.set_xticks(x)
        if not y_lim is None:
            ax.set_ylim(y_lim[0], y_lim[1])
        for i in range(5):
            offset = 2 - i
            bar_vals = loss_Vals[i::5]
            ax.bar(x - width * offset, bar_vals, width, label=types[i])

        ax.legend()

        ax.grid()
        ax.set(ylabel='mean error',
               title=title)

    fig, ax = plt.subplots(1, figsize=(25, 10))

    # create_plot(src_folder + '/mse_order_0_lower.txt', ax, 0)
    # create_plot(src_folder + '/mse_order_1_lower.txt', ax, 1)
    # create_plot(src_folder + '/mse_order_2_lower.txt', ax, 2)
    # create_plot(src_folder + '/mse_order_3_lower.txt', ax, 3)
    # create_plot(src_folder + '/mse_order_0_upper.txt', ax, 4)
    # create_plot(src_folder + '/mse_order_1_upper.txt', ax, 5)
    # create_plot(src_folder + '/mse_order_2_upper.txt', ax, 6)
    # create_plot(src_folder + '/mse_order_3_upper.txt', ax, 7)
    # create_plot(src_folder + '/feet_distances.txt', ax, 8)
    create_plot(src_folder + '/perceptual_loss.txt', ax, 0, y_lim=[0,0.001])

    fig.savefig(target_folder + "/" + name + ".pdf")

def print_dirs(prep, target_folder, name):
    directions = prep.inputs[:,[-14,-12]]

    speeds = np.sqrt(np.square(directions[:, 1]) + np.square(directions[:, 0]))
    n_bins = 20
    fig, ax = plt.subplots(1)
    ax.hist(speeds / prep.target_delta_t, bins=n_bins)
    fig.savefig(target_folder + "/speeds_" + name + ".png")

    directions_ = directions[speeds > 0.01]
    angles = np.arctan2(directions_[:, 1], directions_[:, 0])
    angles = angles + 2 * np.pi
    angles = np.remainder(angles, 2 * np.pi)

    N = 128
    N_ = N * 2
    ranges = np.linspace(0, 2 * np.pi + np.pi / N_, N_ + 1, endpoint=False)


    dir_hist = np.histogram(angles, ranges)[0]
    first_elems = dir_hist[0::2]
    second_elems = np.roll(dir_hist, 1)[0::2]
    dir_hist_end = first_elems + second_elems
    # Compute pie slices
    radii = dir_hist[0]
    width = np.pi / N * 2
    colors = np.random.rand(N, 3)
    theta2 = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    fig, ax = plt.subplots(1, subplot_kw=dict(projection='polar'))

    ax.bar(theta2, np.log10(1 + dir_hist_end), width=width, bottom=0.0, color=[1,0,0], alpha=0.5)

    fig.savefig(target_folder +  "/dirs_log_" + name + ".png")

    fig2, ax_ = plt.subplots(1, subplot_kw=dict(projection='polar'))

    ax_.bar(theta2, dir_hist_end, width=width, bottom=0.0, color=[1,0,0], alpha=0.5)

    fig2.savefig(target_folder +  "/dirs_" + name + ".png")
