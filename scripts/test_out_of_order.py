from Helpers import FileHelpers as fh
from Helpers import StatsPrinter as sp
import numpy as np
from Helpers import DataPreprocessor
from Helpers import ModelWrappers
from sklearn.utils import shuffle
import torch

animation_data_path = "E:/Master/Sorted Mocap/WALKING/walking.npz"
STACKCOUNT = 15
TARGET_FPS = 20
training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)

training_prep.load_np(animation_data_path)

loss_test_wrapper = ModelWrappers.ae_perceptual_loss(training_prep)
loss_test_wrapper.train(0, 600, 0.01)

source_path = "../predictions_print_tests_2"

list_file_name = "animation_list.txt"
animations = fh.get_lines(source_path + "/" + list_file_name)

normal_path = "../test_out/normal"
shuffled_path = "../test_out/shuffled"
rotated_path = "../test_out/rotated"
shuffled_and_rotated_path = "../test_out/rotated&shuffled"

for animation in animations:
    np_file = np.load(animation)

    generated_data=np_file['generated_data']
    reference_data=np_file['reference_data']

    shuffled_reference, shuffled_generated = shuffle(reference_data, generated_data, random_state=42)

    diff_vec = reference_data - generated_data
    mse = np.linalg.norm(diff_vec, axis=2)

    random_vectors = np.random.rand(generated_data.shape[0], generated_data.shape[1], generated_data.shape[2])
    rand_mse = np.linalg.norm(random_vectors, axis=2)

    factor = mse / rand_mse
    random_vectors *= factor[:, :, np.newaxis]

    rotated_generated_vectors = reference_data - random_vectors
    mse2 = np.linalg.norm(reference_data - rotated_generated_vectors, axis=2)

    shuffled_rotated_reference, shuffled_and_rotated_generated = shuffle(reference_data, rotated_generated_vectors, random_state=42)

    joint_names=list(np_file['joint_names'])
    activity=np_file['activity'].item().split("/")[-1]
    delta_t=np_file['delta_t'].item()

    fh.create_folder(normal_path)
    fh.create_folder(shuffled_path)
    fh.create_folder(rotated_path)
    fh.create_folder(shuffled_and_rotated_path)

    sp.print_stats(generated_data, reference_data, joint_names, activity, delta_t, loss_test_wrapper, save_matrices=False, target_path=normal_path)
    sp.print_stats(rotated_generated_vectors, reference_data, joint_names, activity, delta_t, loss_test_wrapper, save_matrices=False, target_path=rotated_path)
    sp.print_stats(shuffled_generated, shuffled_reference, joint_names, activity, delta_t, loss_test_wrapper, save_matrices=False, target_path=shuffled_path)
    sp.print_stats(shuffled_and_rotated_generated, shuffled_rotated_reference, joint_names, activity, delta_t, loss_test_wrapper, save_matrices=False, target_path=shuffled_and_rotated_path)

target_folder = "test_out"
sp.create_summary(normal_path, target_folder, "overview_normal")
sp.create_summary(shuffled_path, target_folder, "overview_shuffled")
sp.create_summary(rotated_path, target_folder, "overview_rotated")
sp.create_summary(shuffled_and_rotated_path, target_folder, "overview_shuffled_and_rotated")
