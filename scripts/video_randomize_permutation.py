import moviepy

from moviepy.editor import *
from Helpers import FileHelpers

def concatenate_animations(videos, image_folder, target):
    clip1 = VideoFileClip(videos[0])
    clip2 = VideoFileClip(videos[1])
    clip3 = VideoFileClip(videos[2])
    clip4 = VideoFileClip(videos[3])
    duration1 = clip1.duration

    clips = []
    for idx in range(videos.__len__()):


        clip = VideoFileClip(videos[idx])
        currduration = clip1.duration
        identifier = ImageClip(image_folder + "Images\\"+ str(idx + 1) + ".png", duration=currduration)
        clip = CompositeVideoClip([clip, identifier])

        if idx == 0:
            pass
        else:
            colorclip = ColorClip(size=(clip1.w, clip1.h), color=[1, 1, 1], duration=1)
            clip = concatenate_videoclips([colorclip, clip])
        clips.append(clip)


    final_clip = concatenate_videoclips(clips)    # Overlay the text clip on the first video clip
    final_clip.write_videofile(target, codec='libx264')


file_path = "E:\\Systemordner\\Dokumente\\Pycharm\\Master\\sparse_body_pose_prediction\\moglow_dropout\\unity_motion_export\\UNTITYEXPORT\\"
image_folder = ""
numbers = [45, 720, 734,  # bwd
           338, 1148, 2112,  # circle
           650, 763, 2308,  # diagonal
           976, 1514, 2016,  # fwd
           12, 13, 772  # sideways
           ]
methods = ["RNN2", "REF", "GLOW", "IK"]

for number in numbers:
    file_names = [file_path + "WALKING_" + method + "_" + str(number) + "_trained_on_WALKING.mp4" for method in methods]
    out_file_path = "shuffled_videos\\"
    FileHelpers.create_folder(out_file_path)
    concatenate_animations([file_names[0],file_names[1],file_names[2],file_names[3]], image_folder, out_file_path + str(number) + "_A.mp4")
    concatenate_animations([file_names[3],file_names[2],file_names[1],file_names[0]], image_folder, out_file_path + str(number) + "_B.mp4")
    concatenate_animations([file_names[2],file_names[3],file_names[0],file_names[1]], image_folder, out_file_path + str(number) + "_C.mp4")
    concatenate_animations([file_names[1],file_names[0],file_names[3],file_names[2]], image_folder, out_file_path + str(number) + "_D.mp4")