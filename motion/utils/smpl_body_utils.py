# Copyright 2024 inria
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import smplx
from pathlib import Path
import torch
import torch.nn.functional as F
import os



mmm_kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                      [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20]]

smplh_kinematic_tree = [[0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20],
                        [9, 14, 17, 19, 21], [0, 1, 4, 7, 10],
                        [0, 2, 5, 8, 11]]
mmm_to_smplh_scaling_factor = 0.75 / 480

colors = {
        "blue": [0, 0, 255, 1],
        "cyan": [0, 255, 255, 1],
        "green": [0, 128, 0, 1],
        "yellow": [255, 255, 0, 1],
        "red": [255, 0, 0, 1],
        "grey": [77, 77, 77, 1],
        "black": [0, 0, 0, 1],
        "white": [255, 255, 255, 1],
        "transparent": [255, 255, 255, 0],
        "magenta": [197, 27, 125, 1],
        'pink': [197, 140, 133, 1],
        "light_grey": [217, 217, 217, 255],
        'yellow_pale': [226, 215, 132, 1],
        }
marker2bodypart = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55],
}

bodypart2color = {
    "head_ids": 'cyan',
    "mid_body_ids": 'blue',
    "left_hand_ids": 'red',
    "right_hand_ids": 'green',
    "left_foot_ids": 'grey',
    "right_foot_ids": 'black',
    "left_toe_ids": 'yellow',
    "right_toe_ids": 'magenta',
    "special": 'light_grey'
}


def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c



#=====================================================================================================
# Get smpl body model from smplx
def get_body_model(model_type, gender, batch_size, device='cpu', ext='pkl'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    mtype = model_type.upper()
    #if gender != 'neutral':
    #    if not isinstance(gender, str):
    #        gender = str(gender.astype(str)).upper()
    #    else:
    #        gender = gender.upper()
    if(model_type=='smplh'):
        ext = 'npz'
    else:
        gender = gender.upper()
        
    body_model_path = os.path.join(os.getcwd(), f'data/models/{model_type}/{mtype}_{gender}.{ext}')
    model_params = dict(model_path=body_model_path, model_type=mtype,
                        gender=gender, ext=ext,
                        use_pca=False,
                        num_pca_comps=12,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        batch_size=batch_size,)
    body_model = smplx.create(**model_params, encoding='latin1').to(device=device)
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model