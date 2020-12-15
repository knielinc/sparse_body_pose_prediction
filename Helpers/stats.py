from glob import glob
from os import listdir
from os.path import isfile, join
from bvhtoolbox import bvh

folders = glob("C:/Users/cknie/Desktop/Sorted Movement/*/")

frameSum = 0
timeSum = 0

for path in folders:
    print(path[("C:/Users/cknie/Desktop/Sorted Movement/").__len__():-1] + " :")
    allfiles = [f for f in listdir(path) if isfile(join(path, f))]

    time = 0
    frames = 0
    for file in allfiles:
        with open(path + file) as f:
            mocap = bvh.Bvh(f.read())
            frames += mocap.nframes
            time += mocap.nframes * mocap.frame_time
    print("\t frames: " + str(frames) + "\n\t totaltime: " + str(time))
    frameSum += frames
    timeSum += time

print("\n============================================================\n\tSummary\n============================================================")
print("\t frames: " + str(frameSum) + "\n\t totaltime: " + str(timeSum))
