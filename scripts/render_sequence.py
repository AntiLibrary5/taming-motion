#
# Copyright 2024 adakri
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


import argparse
import os
import sys
sys.path.append('./')
import numpy as np
import motion.utils.geometry_utils as geometry
import torch
import motion.utils.smpl_body_utils as smpl_body_utils
import motion.render.mesh_viz as mesh_viz
from motion.utils.smpl_from_joints import joints2smpl
from motion.utils.rotation2xyz import Rotation2xyz
from motion.render.mesh_viz import render_motion_sequence

device='cpu'

def main(dataset_path, sequence_id, output_folder, description, threads, debug, device='cpu',):
    # Read motion and associated caption
    assert os.path.exists(dataset_path)
    motion_vec_path = os.path.join(dataset_path, "new_joint_vecs", f"{sequence_id.zfill(6)}.npy") #Nx263
    joints_vec_path = os.path.join(dataset_path, "new_joints", f"{sequence_id.zfill(6)}.npy") #Nx22
    caption_path = os.path.join(dataset_path, "texts", sequence_id.zfill(6)+ ".txt")
    # Read
    motion_vec = torch.tensor(np.load(motion_vec_path), device=device)
    joints_vec = np.load(joints_vec_path)
    caption = open(caption_path).readlines()
    
    render_motion_sequence(motion_vec, joints_vec, caption, sequence_id, output_folder, description, threads, True, device)
    
    return



if __name__ == "__main__":
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-p", type=str, required=True)
    parser.add_argument("--sequence_id", "-sid", type=str, required=True)
    parser.add_argument("--output_folder", "-o", type=str, default='./output', required=False)
    parser.add_argument("--description", "-d", type=str, default="", required=False,
                        help="Optional description to add to the output file.")
    
    parser.add_argument("--threads", "-t", type=int, default=4, required=False)
     
    parser.add_argument("--debug", action="store_true", required=False)
    args = parser.parse_args()
    
    main(**vars(args), device=device)  
    