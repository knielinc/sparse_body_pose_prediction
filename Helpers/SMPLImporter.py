import sys, os
import torch
import numpy as np
import os
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'SMPLHModels')
COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JOINT_COUNT = 22

def get_joint_names():
    return ['Hips',             # 0
            'LeftUpLeg',        # 1
            'RightUpLeg',       # 2
            'Spine',            # 3
            'LeftLeg',          # 4
            'RightLeg',         # 5
            'Spine1',           # 6
            'LeftFoot',         # 7
            'RightFoot',        # 8
            'Spine2',           # 9
            'LeftToeBase',      # 10
            'RightToeBase',     # 11
            'Neck',             # 12
            'LeftShoulder',     # 13
            'RightShoulder',    # 14
            'Head',             # 15
            'LeftArm',          # 16
            'RightArm',         # 17
            'LeftForeArm',      # 18
            'RightForeArm',     # 19
            'LeftHand',         # 20
            'RightHand',        # 21
            'LeftHandIndex1',   # 22
            'RightHandIndex1']  # 23

def get_dependencies():
    return np.array([[0, -1],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 1],
            [5, 2],
            [6, 3],
            [7, 4],
            [8, 5],
            [9, 6],
            [10, 7],
            [11, 8],
            [12, 9],
            [13, 9],
            [14, 9],
            [15, 12],
            [16, 13],
            [17, 14],
            [18, 16],
            [19, 17],
            [20, 18],
            [21, 19],
            ])


class SMPLImporter:
    def __init__(self, file_name):
        FRAMES, FRAME_TIME, joint_names, dependencies, zipped_global_positions = load_file(file_name)
        self.joint_names = joint_names
        self.dependencies = dependencies
        # self.zipped_global_quat_rotations = zipped_global_quat_rotations
        # self.zipped_local_quat_rotations = zipped_local_quat_rotations
        self.zipped_global_positions = zipped_global_positions
        # self.zipped_local_positions = zipped_local_positions
        self.frame_time = FRAME_TIME
        self.nr_of_frames = FRAMES

def load_file(file_name):
    npz_bdata_path = file_name
    bdata = np.load(npz_bdata_path)

    gender = bdata['gender']

    bm_path = MODELS_FOLDER + "/" + str(gender) + '/model.npz'

    num_betas = 10 # number of body parameters
    model_type = 'smplh'

    bm = BodyModel(bm_path=bm_path, num_betas=num_betas, model_type=model_type).to(COMP_DEVICE)
    faces = c2c(bm.f)

    # print('Data keys available:%s'%list(bdata.keys()))
    # print('Vector poses has %d elements for each of %d frames.' % (bdata['poses'].shape[1], bdata['poses'].shape[0]))
    # print('Vector dmpls has %d elements for each of %d frames.' % (bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
    # print('Vector trams has %d elements for each of %d frames.' % (bdata['trans'].shape[1], bdata['trans'].shape[0]))
    # print('Vector betas has %d elements constant for the whole sequence.'%bdata['betas'].shape[0])
    # print('The subject of the mocap sequence is %s.'%bdata['gender'])

    FRAMES = bdata['poses'].shape[0]
    FRAME_TIME = 1.0 / bdata['mocap_framerate'].item()

    joint_names = get_joint_names()[:JOINT_COUNT]
    dependencies = get_dependencies()

    zipped_global_positions = []

    root_orient_allframes = torch.Tensor(bdata['poses'][:, :3]).to(COMP_DEVICE)  # controls the global root orientation
    pose_body_allframes = torch.Tensor(bdata['poses'][:, 3:66]).to(COMP_DEVICE)  # controls the body
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(COMP_DEVICE)
    trans_allframes = bdata['trans']

    for fId in range(FRAMES):
        root_orient = root_orient_allframes[fId:fId+1]
        pose_body = pose_body_allframes[fId:fId+1]

        body = bm(root_orient=root_orient, pose_body=pose_body, betas=betas)
        joints = (c2c(body.Jtr[0]))[:JOINT_COUNT, :]
        zipped_global_positions.append(joints + trans_allframes[fId])
        # if (fId / FRAMES * 100) % 10 == 0:
        #     print("processing smplh file.. " + str(fId / FRAMES * 100) + "%")

    zipped_global_positions = np.array(zipped_global_positions)[:, :, [0, 2, 1]]
    return FRAMES, FRAME_TIME, joint_names, dependencies, zipped_global_positions
