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

device='cuda:0'

def main(dataset_path, sequence_id, output_folder, description, threads, debug, device='cpu'):
    # Read motion and associated caption
    assert os.path.exists(dataset_path)
    motion_vec_path = os.path.join(dataset_path, "new_joint_vecs", sequence_id.zfill(6)+ ".npy") #Nx263
    joints_vec_path = os.path.join(dataset_path, "new_joints", sequence_id.zfill(6)+ ".npy") #Nx22
    caption_path = os.path.join(dataset_path, "texts", sequence_id.zfill(6)+ ".txt")
    # Read
    motion_vec = torch.tensor(np.load(motion_vec_path), device=device)[:3,:]
    joints_vec = torch.tensor(np.load(joints_vec_path), device=device)[:3,:]
    caption = open(caption_path).readlines()
    
    # Rendering
    frames = joints_vec.shape[0]
    MINS = joints_vec.min(axis=0)[0].min(axis=0)[0]
    MAXS = joints_vec.max(axis=0)[0].max(axis=0)[0]

    #height_offset = MINS[1]
    #joints_vec[:, :, 1] -= height_offset
    trajec = joints_vec[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames,)
    rot2xyz = Rotation2xyz(device=torch.device(device))
    faces = rot2xyz.smpl_model.faces

    print(f'Running SMPLify, it may take a few minutes.')
    motion_tensor, opt_dict = j2s.joint2smpl(joints_vec)  # [nframes, njoints, 3]

    vertices = rot2xyz(torch.tensor(motion_tensor, device=device).clone(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True,)
    # Generate animation
    mesh_viz.visualize_meshes(
        vertices.squeeze(0).permute(-1,0,1).detach().cpu().numpy(), save_path=os.path.join(output_folder, f"test.gif"), fig_label=caption)
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
    