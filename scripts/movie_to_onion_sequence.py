# Opens the Video file
import cv2
# 13_A.mp4
import numpy as np
nr = 1514

def make_alpha(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,20,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst

folder = "E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT"

cap_ref = cv2.VideoCapture(folder  + "/WALKING_" + "REF_" + str(nr) + "_trained_on_WALKING" + ".mp4")
cap_ff = cv2.VideoCapture(folder  + "/WALKING_" + "FF_" + str(nr) + "_trained_on_WALKING" + ".mp4")
cap_rnn = cv2.VideoCapture(folder  + "/WALKING_" + "RNN2_" + str(nr) + "_trained_on_WALKING"  + ".mp4")
cap_glow = cv2.VideoCapture(folder  + "/WALKING_" + "GLOW_" + str(nr) + "_trained_on_WALKING"+ ".mp4")
cap_ik = cv2.VideoCapture(folder  + "/WALKING_" + "IK_" + str(nr) + "_trained_on_WALKING" + ".mp4")

# cap_ff = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_FF_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_rnn = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_RNN2_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_glow = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_GLOW_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_ik = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_IK_' + str(nr)+ '_trained_on_WALKING.mp4')

# cap_ref = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_REF_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_ff = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_FF_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_rnn = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_RNN2_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_glow = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_GLOW_' + str(nr)+ '_trained_on_WALKING.mp4')
# cap_ik = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_IK_' + str(nr)+ '_trained_on_WALKING.mp4')
i = 0
images = []
framerate = 25
while (cap_ff.isOpened()):
    if i > 590:
        break
    # ret, frame_ref  = cap_ref.read()
    ret, frame_ff   = cap_ff.read()
    ret, frame_rnn  = cap_rnn.read()
    ret, frame_ref = cap_ref.read()
    ret, frame_ik = cap_ik.read()
    ret, frame_glow = cap_glow.read()
    # ret, frame_ik   = cap_ik.read()
    if ret == False:
        break

    if i % framerate == 0 and i > 160:
        images.append(cv2.vconcat([frame_ref, frame_ff, frame_rnn, frame_glow, frame_ik]))
    i += 1

img_seq =  make_alpha(images[0]).astype(np.float32)
first_image = img_seq.copy()
last_image = img_seq.copy()


for img_ in images[1:]:
    # img_seq = cv2.addWeighted(img_seq,.8,img_,.2,0)
    # img_seq = cv2.add(img_seq, img_.astype(np.float32))
    # img2gray = cv2.cvtColor(img_seq, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    #
    # img_seq = cv2.bitwise_and(img_seq, img_seq, mask=mask)
    # img2_fg = cv2.bitwise_and(img_, img_, mask=mask_inv)
    last_image = make_alpha(img_).astype(np.float32).copy()
    # cv2.imshow("kekw", last_image)
    img_seq = cv2.add(img_seq, last_image)
    # img_seq = cv2.addWeighted(img_seq,.9,img_,.2,0)
    # img2_fg = cv2.addWeighted(img2_fg,.8,img_,.2,0)

    # img_seq = cv2.add(img_seq,img2_fg)

# factor = 1./images.__len__()
img_seq = cv2.addWeighted(img_seq,.5,img_seq,0,0)
img_seq = cv2.add(img_seq,first_image)
img_seq = cv2.add(img_seq,last_image)
# img_seq = cv2.vconcat(images[:9])
# img_seq_cropped = img_seq[:,380:-380]

cv2.imwrite("ref_" + str(nr) + '.png', img_seq)
# cv2.imwrite("test_concat_crop" + '.jpg', img_seq_cropped)

# cap_ref.release()
cap_ff.release()
cap_rnn.release()
cap_glow.release()
cap_ik.release()
cv2.destroyAllWindows()


# 1280×720
# 1280 - 720 = 560 -> 380