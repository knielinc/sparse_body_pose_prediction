import numpy as np

TAB = "\t"

def print_stats(generated_data, reference_data, joint_names, activity):
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
