from Helpers import DataPreprocessor
from Helpers import ModelWrappers
from Helpers import Models
import torch
import sys
import time

from Helpers import FileHelpers

take_last = lambda x: og_folder + "/" + x.split('/')[-1].split('.')[0]

og_folder = "TEMP"

FileHelpers.create_folder(og_folder)
FileHelpers.create_folder(og_folder + "/unity_motion_export")
FileHelpers.create_folder(og_folder + "/stats")
FileHelpers.create_folder(og_folder + "/videos_out")

PATH_PREFIX = "E:/Master/Sorted Mocap/WALKING/WALKING_"
numbers = [45, 720, 734,  # bwd
           338, 1148, 2112,  # circle
           650, 763, 2308,  # diagonal
           976, 1514, 2016,  # fwd
           12, 13, 772  # sideways
           ]
eval_files = [PATH_PREFIX + str(elem) + ".npz" for elem in numbers]


STACKCOUNT = 15
TARGET_FPS = 20

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5, use_weighted_sampling=True)
training_prep.load_scalers("saved_params")

#rnn2
rnn_wrapper = ModelWrappers.rnn_wrapper_2(training_prep)

# rnn_wrapper.train(80, FF_BATCH_SIZE, 0.0001)
rnn_wrapper.load_model('saved_params/rnn_test_save')

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    eval_prep.finalize()

    rnn_wrapper.predict(eval_prep)

    total_time = rnn_wrapper.last_inference_time

    rnn_wrapper.save_prediction(take_last(file) + "_RNN", None)
