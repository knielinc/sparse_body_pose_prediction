from Helpers import DataPreprocessor
from Helpers import ModelWrappers
from Helpers import Models
import torch
import sys
from Helpers import FileHelpers

og_folder = "moglow_dropout"

FileHelpers.create_folder(og_folder)
FileHelpers.create_folder(og_folder + "/unity_motion_export")
FileHelpers.create_folder(og_folder + "/stats")
FileHelpers.create_folder(og_folder + "/videos_out")

def test_folder_func(folder_name, FF_BATCH_SIZE):
    STACKCOUNT = 15
    TARGET_FPS = 20
    take_last = lambda x : og_folder + "/" + x.split('/')[-1].split('.')[0] + "_trained_on_" + folder_name.split("/")[-2]

    # eval_files = [  "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/greeting-01-hello-hamada_poses.npz",
    #                 "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kudo/jump-12-boxer step-kudo_poses.npz",
    #                 "E:/Master/Converted Mocap/KIT/576/MarcusS_AdrianM11_poses.npz",
    #                 "E:/Master/Converted Mocap/KIT/513/balance_on_beam06_poses.npz",
    #                 "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/gesture_etc-14-apologize-hamada_poses.npz",
    #                 "E:/Master/Converted Mocap/Eyes_Japan_Dataset/kanno/walk-01-normal-kanno_poses.npz",
    #                 "E:/Master/Converted Mocap/Eyes_Japan_Dataset/takiguchi/pose-10-recall blackmagic-takiguchi_poses.npz",
    #                 "E:/Master/Converted Mocap/TotalCapture/s1/freestyle2_poses.npz",
    #                 "E:/Master/Converted Mocap/Eyes_Japan_Dataset/hamada/accident-02-dodge fast-hamada_poses.npz",
    #                 "E:/Master/Converted Mocap/BMLhandball/S07_Expert/Trial_upper_left_right_003_poses.npz"]


    # eval_files_old = [ "E:/Master/Sorted Mocap/WALKING/WALKING_2265.npz",
    #                "E:/Master/Sorted Mocap/BASKETBALL/BASKETBALL_10.npz",
    #                "E:/Master/Sorted Mocap/BOXING/BOXING_64.npz",
    #                "E:/Master/Sorted Mocap/THROWING/THROWING_58.npz",
    #                "E:/Master/Sorted Mocap/INTERACTION/INTERACTION_1534.npz"
    #                ]
    PATH_PREFIX = "E:/Master/Sorted Mocap/WALKING/WALKING_"
    numbers = [45,720,734, #bwd
               338,1148,2112, #circle
               650,763,2308, #diagonal
               976, 1514, 2016, #fwd
               12, 13, 772 #sideways
               ]
    # numbers = []

    eval_files = [PATH_PREFIX + str(elem) + ".npz" for elem in numbers]
    # eval_files = [ "E:/Master/Sorted Mocap/WALKING/WALKING_42.npz",
    #                "E:/Master/Sorted Mocap/WALKING/WALKING_360.npz",
    #                "E:/Master/Sorted Mocap/WALKING/WALKING_420.npz",
    #                "E:/Master/Sorted Mocap/WALKING/WALKING_1337.npz",
    #                "E:/Master/Sorted Mocap/WALKING/WALKING_2265.npz",
    #                ]

    training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5, use_weighted_sampling=True)

    training_prep.append_folder(folder_name, eval_files, mirror=True, reverse=True)
    training_prep.finalize()
    training_prep.save(folder_name + "walking_augmented_non_weighted_non_augmented.npz")
    # training_prep.load_np(folder_name + "walking_augmented_3.npz")
    # from Helpers import StatsPrinter
    # StatsPrinter.print_dirs(training_prep)

    # training_prep.load_np(folder_name + "walking_augmented_2.npz")
    # training_prep.load_np(folder_name + "walking_2.npz")

    # training_prep.load_np(folder_name + "combined.npz")
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

    # training_prep.append_file("E:/Master/Sorted Mocap/WALKING/WALKING_2265.npz")
    # training_prep.finalize()
    # for idx in range(1,eval_files.__len__()):
    #     training_prep.append_file(eval_files[idx])
    # training_prep.append_file(eval_files[1])


    # loss_test_wrapper = ModelWrappers.ae_perceptual_loss(training_prep)
    # loss_test_wrapper.train(400, 600, 0.01)
    # gan_wrapper = None
    #
    gan_wrapper = ModelWrappers.gan_wrapper(training_prep)
    gan_wrapper.train(1, FF_BATCH_SIZE, 0.0001)
    # GLOW
    glow_wrapper = ModelWrappers.glow_wrapper(training_prep)
    glow_wrapper.train(60, 180)

    for file in eval_files:
        eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
        eval_prep.append_file(file)
        eval_prep.finalize()
        glow_wrapper.predict(eval_prep)
        glow_wrapper.save_prediction(take_last(file) + "_GLOW", gan_wrapper)

    torch.cuda.empty_cache()

    # FF
    ff_wrapper = ModelWrappers.ff_wrapper(training_prep)
    ff_wrapper.train(120, FF_BATCH_SIZE, 0.0001)

    for file in eval_files:
        eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
        eval_prep.append_file(file)
        eval_prep.finalize()
        ff_wrapper.predict(eval_prep)
        ff_wrapper.save_prediction(take_last(file) + "_FF", gan_wrapper)

    torch.cuda.empty_cache()

    #rnn2
    rnn_wrapper = ModelWrappers.rnn_wrapper_2(training_prep)
    rnn_wrapper.train(80, FF_BATCH_SIZE, 0.0001)
    for file in eval_files:
        eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
        eval_prep.append_file(file)
        eval_prep.finalize()
        rnn_wrapper.predict(eval_prep)
        rnn_wrapper.save_prediction(take_last(file) + "_RNN2", gan_wrapper)

    # #VAE
    # vae_wrapper = ModelWrappers.vae_wrapper(training_prep)
    # vae_wrapper.train(5, 2, FF_BATCH_SIZE, 0.0001)
    #
    # for file in eval_files:
    #     eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    #     eval_prep.append_file(file)
    #     vae_wrapper.predict(eval_prep)
    #     vae_wrapper.save_prediction(take_last(file) + "_VAE", loss_test_wrapper)
    # RNN
    # rnn_wrapper = ModelWrappers.rnn_wrapper(training_prep)
    # rnn_wrapper.train(50, 90, FF_BATCH_SIZE, 0.0001)
    #
    # for file in eval_files:
    #     eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    #     eval_prep.append_file(file)
    #     rnn_wrapper.predict(eval_prep)
    #     rnn_wrapper.save_prediction(take_last(file) + "_RNN3", gan_wrapper)


    # gan_wrapper = ModelWrappers.gan_wrapper(training_prep)
    # gan_wrapper.train(100, FF_BATCH_SIZE, 0.0001)
    # for file in eval_files:
    #     eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
    #     eval_prep.append_file(file)
    #     eval_prep.finalize()
    #     gan_wrapper.predict(eval_prep)
    #     gan_wrapper.save_prediction(take_last(file) + "_GAN", None)

perceptual_loss_file_name = og_folder + '/perceptual_loss.txt'

open(perceptual_loss_file_name, 'w').close()

# test_folder_func("E:/Master/Sorted Mocap/BASKETBALL/", 100)
# test_folder_func("E:/Master/Sorted Mocap/BOXING/", 100)
test_folder_func("E:/Master/Sorted Mocap/WALKING/", 4000)
# test_folder_func("E:/Master/Sorted Mocap/THROWING/", 4000)
# test_folder_func("E:/Master/Sorted Mocap/INTERACTION/", 4000)

