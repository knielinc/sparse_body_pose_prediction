from Helpers import DataPreprocessor
from Helpers import ModelWrappers

STACKCOUNT = 10
TARGET_FPS = 20

eval_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 1)
eval_prep.append_file("C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/17_03.npz")

training_prep = DataPreprocessor.ParalellMLPProcessor(STACKCOUNT, 1.0 / TARGET_FPS, 5)
training_prep.append_folder("C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/", ["C:/Users/cknie/Desktop/convertedMocapData_bvh/Walking/17_03.npz"])

glow_wrapper = ModelWrappers.ff_wrapper(training_prep)

glow_wrapper.train(150, 20000, 0.0001)
glow_wrapper.predict(eval_prep)
glow_wrapper.save_prediction()
