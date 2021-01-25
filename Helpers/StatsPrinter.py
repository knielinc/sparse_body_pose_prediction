import numpy as np
import matplotlib
import matplotlib.pyplot as plt
TAB = "\t"

def save_feet_distance(generated_data, reference_data, joint_names, activity):
    t = np.arange(0.0, generated_data.shape[0], 1)

    rl = joint_names.index('l_foot')
    ll = joint_names.index('r_foot')

    sample_feet_dists       = np.linalg.norm(generated_data[:,ll,:] - generated_data[:,rl,:], axis=1)
    reference_feet_dists    = np.linalg.norm(reference_data[:,ll,:] - reference_data[:,rl,:], axis=1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, sample_feet_dists, color='green', label='estimated')
    ax.plot(t, reference_feet_dists, color='red', label='ground_truth')

    ax.legend()

    ax.set(xlabel='time (s)', ylabel='distance (m)',
           title='Distance between Feet over Time')
    ax.grid()

    fig.savefig(activity + ".png")

def print_stats(generated_data, reference_data, joint_names, activity):
    save_feet_distance(generated_data, reference_data, joint_names, activity)
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
