from Helpers import DataPreprocessor
from Helpers import ModelWrappers
import sys


STACKCOUNT = 15
TARGET_FPS = 20
take_last = lambda x : "predictions/" + x.split('/')[-1].split('.')[0]

eval_files = [  "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/greeting-01-hello-hamada_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kudo/jump-12-boxer step-kudo_poses.npz",
                "E:/Master/Converted Mocap/KIT/576/MarcusS_AdrianM11_poses.npz",
                "E:/Master/Converted Mocap/KIT/513/balance_on_beam06_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/gesture_etc-14-apologize-hamada_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kanno/walk-01-normal-kanno_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/takiguchi/pose-10-recall blackmagic-takiguchi_poses.npz",
                "E:/Master/Converted Mocap/TotalCapture/s1/freestyle2_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/accident-02-dodge fast-hamada_poses.npz",
                "E:/Master/Converted Mocap/BMLhandball/S07_Expert/Trial_upper_left_right_003_poses.npz"]

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)

# training_prep.append_subfolders("E:/Master/Converted Mocap/ACCAD", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLhandball", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLmovi", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/DFaust_67", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/EKUT", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/Eyes_Japan_Dataset", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/HumanEva", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/Kit", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_HDM05", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Moca/MPI_Limits", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_mosh", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/SFU", eval_files)
# training_prep.append_subfolders("E:/Master/Converted Mocap/TotalCapture", eval_files)


training_prep.load_np("E:/Master/Converted Mocap/combined_without_eval.npz")
# training_prep.append_file(eval_files[1])
#FF
ff_wrapper = ModelWrappers.ff_wrapper(training_prep)
ff_wrapper.train(100, 20000, 0.0001)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    ff_wrapper.predict(eval_prep)
    ff_wrapper.save_prediction(take_last(file) + "_FF")
#VAE
vae_wrapper = ModelWrappers.vae_wrapper(training_prep)
vae_wrapper.train(5, 2, 20000, 0.0001)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    vae_wrapper.predict(eval_prep)
    vae_wrapper.save_prediction(take_last(file) + "_VAE")
#RNN
rnn_wrapper = ModelWrappers.rnn_wrapper(training_prep)
rnn_wrapper.train(100, 90, 20000, 0.0001)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    rnn_wrapper.predict(eval_prep)
    rnn_wrapper.save_prediction(take_last(file) + "_RNN")
# GLOW
glow_wrapper = ModelWrappers.glow_wrapper(training_prep)
glow_wrapper.train(100, 50)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    glow_wrapper.predict(eval_prep)
    glow_wrapper.save_prediction(take_last(file) + "_GLOW")




