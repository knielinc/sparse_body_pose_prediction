from moviepy.editor import *

def concatenate_animations(video1, video2, target):
    clip1 = VideoFileClip(video1)
    clip2 = VideoFileClip(video2)
    final_clip = clips_array([[clip1, clip2]])
    # Overlay the text clip on the first video clip
    final_clip.write_videofile(target, codec='libx264')