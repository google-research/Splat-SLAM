# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = torch.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def update_cam(cfg):
    """
    Update the camera intrinsics according to the pre-processing config,
    such as resize or edge crop
    """
    # resize the input images to crop_size(variable name used in lietorch)
    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy = cfg['cam']['cx'], cfg['cam']['cy']

    h_edge, w_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']

    fx = fx * (W_out + w_edge * 2) / W
    fy = fy * (H_out + h_edge * 2) / H
    cx = cx * (W_out + w_edge * 2) / W
    cy = cy * (H_out + h_edge * 2) / H
    H, W = H_out, W_out

    cx = cx - w_edge
    cy = cy - h_edge
    return H,W,fx,fy,cx,cy    


@torch.no_grad()
def align_scale_and_shift(prediction, target, weights):

    '''
    weighted least squares problem to solve scale and shift: 
        min sum{ 
                  weight[i,j] * 
                  (prediction[i,j] * scale + shift - target[i,j])^2 
               }

    prediction: [B,H,W]
    target: [B,H,W]
    weights: [B,H,W]
    '''

    if weights is None:
        weights = torch.ones_like(prediction).to(prediction.device)
    if len(prediction.shape)<3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)  
    a_00 = torch.sum(weights * prediction * prediction, dim=[1,2])
    a_01 = torch.sum(weights * prediction, dim=[1,2])
    a_11 = torch.sum(weights, dim=[1,2])
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(weights * prediction * target, dim=[1,2])
    b_1 = torch.sum(weights * target, dim=[1,2])
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b            
    det = a_00 * a_11 - a_01 * a_01
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    error = (scale[:,None,None]*prediction+shift[:,None,None]-target).abs()
    masked_error = error*weights
    error_sum = masked_error.sum(dim=[1,2])
    error_num = weights.sum(dim=[1,2])
    avg_error = error_sum/error_num

    return scale,shift,avg_error