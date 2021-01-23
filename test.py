from Helpers import DataPreprocessor
from Helpers import ModelWrappers

STACKCOUNT = 15
TARGET_FPS = 20
take_last = lambda x : x.split('/')[-1]

eval_files = [  "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/greeting-01-hello-hamada_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kudo/jump-12-boxer step-kudo_poses.npz",
                "E:/Master/Converted Mocap/MPI_HDM05/dg/HDM_dg_03-01_03_120_poses.npz.npz",
                "E:/Master/Converted Mocap/MPI_HDM05/tr/HDM_tr_04-01_01_120_poses.npz",
                "E:/Master/Converted Mocap/KIT/576/MarcusS_AdrianM11_poses.npz",
                "E:/Master/Converted Mocap/SFU/0015/0015_BasicKendo001_poses.npz",
                "E:/Master/Converted Mocap/KIT/513/balance_on_beam06_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/gesture_etc-14-apologize-hamada_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kanno/walk-01-normal-kanno_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/takiguchi/pose-10-recall blackmagic-takiguchi_poses.npz",
                "E:/Master/Converted Mocap/TotalCapture/s1/freestyle2_poses.npz",
                "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/accident-02-dodge fast-hamada_poses.npz",
                "E:/Master/Converted Mocap/BMLhandball/S07_Expert/Trial_upper_left_right_003_poses.npz"]

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)

# training_prep.append_subfolders("E:/Master/Converted Mocap/ACCAD")
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLhandball")
# training_prep.append_subfolders("E:/Master/Converted Mocap/BMLmovi")
# training_prep.append_subfolders("E:/Master/Converted Mocap/DFaust_67")
# training_prep.append_subfolders("E:/Master/Converted Mocap/EKUT")
# training_prep.append_subfolders("E:/Master/Converted Mocap/Eyes_Japan_Dataset")
# training_prep.append_subfolders("E:/Master/Converted Mocap/HumanEva")
# training_prep.append_subfolders("E:/Master/Converted Mocap/Kit")
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_HDM05")
# training_prep.append_subfolders("E:/Master/Converted Moca/MPI_Limits")
# training_prep.append_subfolders("E:/Master/Converted Mocap/MPI_mosh")
# training_prep.append_subfolders("E:/Master/Converted Mocap/SFU")
# training_prep.append_subfolders("E:/Master/Converted Mocap/TotalCapture")

training_prep.load_np("E:/Master/Converted Mocap/combined.npz")

#FF
ff_wrapper = ModelWrappers.ff_wrapper(training_prep)
ff_wrapper.train(100, 10000, 0.0001)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    ff_wrapper.predict(eval_prep)
    ff_wrapper.save_prediction(take_last(file) + "_FF")
#VAE
vae_wrapper = ModelWrappers.vae_wrapper(training_prep)
vae_wrapper.train(100, 90, 20000, 0.0001)

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
#GLOW
glow_wrapper = ModelWrappers.glow_wrapper(training_prep)
glow_wrapper.train(100, 90, 50, 0.0001)

for file in eval_files:
    eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    eval_prep.append_file(file)
    glow_wrapper.predict(eval_prep)
    glow_wrapper.save_prediction(take_last(file) + "_GLOW")




