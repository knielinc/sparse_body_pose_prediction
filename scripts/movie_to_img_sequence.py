# Opens the Video file
import cv2
# 13_A.mp4
nr = 12

cap_ref = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_REF_' + str(nr)+ '_trained_on_WALKING.mp4')
cap_ff = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_FF_' + str(nr)+ '_trained_on_WALKING.mp4')
cap_rnn = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_RNN2_' + str(nr)+ '_trained_on_WALKING.mp4')
cap_glow = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_GLOW_' + str(nr)+ '_trained_on_WALKING.mp4')
cap_ik = cv2.VideoCapture('E:/Systemordner/Dokumente/Pycharm/Master/sparse_body_pose_prediction/moglow_dropout/unity_motion_export/UNTITYEXPORT/WALKING_IK_' + str(nr)+ '_trained_on_WALKING.mp4')
i = 0
images = []
framerate = 10
while (cap_ref.isOpened()):
    if i > framerate * 9:
        break
    ret, frame_ref  = cap_ref.read()
    ret, frame_ff   = cap_ff.read()
    ret, frame_rnn  = cap_rnn.read()
    ret, frame_glow = cap_glow.read()
    ret, frame_ik   = cap_ik.read()
    if ret == False:
        break

    if i % framerate == 0:
        images.append(cv2.hconcat([frame_ref, frame_ff, frame_rnn, frame_glow, frame_ik]))
    i += 1

img_seq = cv2.vconcat(images[:9])
# img_seq_cropped = img_seq[:,380:-380]

cv2.imwrite("ref_" + str(nr) + '.jpg', img_seq)
# cv2.imwrite("test_concat_crop" + '.jpg', img_seq_cropped)

cap_ref.release()
cap_ff.release()
cap_rnn.release()
cap_glow.release()
cap_ik.release()
cv2.destroyAllWindows()


# 1280×720
# 1280 - 720 = 560 -> 380