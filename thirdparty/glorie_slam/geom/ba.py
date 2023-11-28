# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lietorch
import torch
import torch.nn.functional as F
from .chol import block_solve, schur_solve
import thirdparty.glorie_slam.geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

def wq_retr(wqs, dwq, ii):
    ii = ii.to(device=dwq.device)
    return wqs + scatter_sum(dwq, ii, dim=1, dim_size=wqs.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

@torch.no_grad()
def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, 
       sensor_disps=None, lm=0.0001, ep=0.1, alpha=0.05, fixedp=1, rig=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1) #[B,N,2*ht*wd,D]
    w = .001 * (valid * weight).view(B, N, -1, 1) #[B,N,2*ht*wd,D]


    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)        #[B,N,2*ht*wd,D]
    Jj = Jj.reshape(B, N, -1, D)        #[B,N,2*ht*wd,D]
    wJiT = (w * Ji).transpose(2,3)      #[B,N,D,2*ht*wd]
    wJjT = (w * Jj).transpose(2,3)      #[B,N,D,2*ht*wd]

    Jz = Jz.reshape(B, N, ht*wd, -1)    #[B,N,ht*wd,2]

    Hii = torch.matmul(wJiT, Ji)        #[B,N,D,D]
    Hij = torch.matmul(wJiT, Jj)        #[B,N,D,D]
    Hji = torch.matmul(wJjT, Ji)        #[B,N,D,D]
    Hjj = torch.matmul(wJjT, Jj)        #[B,N,D,D]
 
    vi = torch.matmul(wJiT, r).squeeze(-1) #[B,N,D]
    vj = torch.matmul(wJjT, r).squeeze(-1) #[B,N,D]

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1) #[B,N,D,ht*wd]
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1) #[B,N,D,ht*wd]

    w = w.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    r = r.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    wk = torch.sum(w*r*Jz, dim=-1) #[B,N,ht*wd]
    Ck = torch.sum(w*Jz*Jz, dim=-1) #[B,N,ht*wd]


    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = torch.div(P,rig,rounding_mode="trunc")-fixedp
    ii = torch.div(ii,rig,rounding_mode="trunc")-fixedp
    jj = torch.div(jj,rig,rounding_mode="trunc")-fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)            #[B,P*P,D,D]

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)             #[B,P*M,D,ht*wd]

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)                    #[B,P,D]

    C = safe_scatter_add_vec(Ck, kk, M)                    #[B,M,ht*wd]

    # C = C + eta.view(*C.shape) #+ 1e-7

    w = safe_scatter_add_vec(wk, kk, M)  #[B,M,ht*wd]
    if sensor_disps is None:
        C = C + eta.view(*C.shape) #+ 1e-7
    else:
        m = (sensor_disps[:,kx]>0).float().view(B,M,ht*wd)     #[B,M,ht*wd]
        C = C + m*alpha + (1-m)*eta.view(*C.shape)             #[B,M,ht*wd]
        w = w - m*alpha*(disps[:,kx]-sensor_disps[:,kx]).view(B,M,ht*wd)  #[B,M,ht*wd]


    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w, ep, lm)
    # dx [B,P,D]
    # dz [B,M,ht*wd]

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps





@torch.no_grad()
def BA_with_scale_shift(target, weight, eta, poses, disps, intrinsics, ii, jj, 
       mono_disps, scales=None, shifts=None, 
       valid_depth_mask=None, ignore_frames=0,
       lm=0.0001, ep=0.1, alpha=1.0, fixedp=1, rig=1):
    """ optimize disparities (disp), scales (w) and shifts (q) together, eq.17 in the paper,
        math details can be found in the supplementary
    """
    device = ii.device
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]
    sqrt_alpha = torch.tensor(alpha).sqrt().to(device)
    ll = torch.arange(M,device=device)
    wqs = torch.stack([scales,shifts],dim=2)         #[B,P,2]

    ignore_mask = kx<ignore_frames
    invalid_mask = (mono_disps[:,kx]<1e-6).view(B,M,ht*wd)    #[B,M,ht*wd]
    invalid_mask[:,ignore_mask] = True

    valid_depth_mask = valid_depth_mask[:,kx].view(B,M,ht*wd)
    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)           #[B,N,ht*wq*2,1]
    r_depth = sqrt_alpha * (disps[:,kx]-(scales[:,kx,None,None]*mono_disps[:,kx]+shifts[:,kx,None,None])).view(B,M,ht*wd,1)

    w = .001 * (valid * weight).view(B, N, -1, 1)

    sqrt_alpha = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    sqrt_alpha[valid_depth_mask] *= 10

    J_d = torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha
    J_scale = -mono_disps[:,kx].clone().view(B,M,ht*wd,1) * sqrt_alpha#[B,M,ht*wd,1]
    J_shift = -torch.ones(B,M,ht*wd,1).float().to(device) * sqrt_alpha#[B,M,ht*wd,1]

    J_d[invalid_mask*valid_depth_mask] = 0
    J_scale[invalid_mask] = 0
    J_shift[invalid_mask] = 0

    J_wq = torch.cat([J_scale,J_shift],dim=3)         #[B,M,ht*wd,2]
    J_wq_T = J_wq.transpose(2,3)         #[B,M,2,ht*wd]
    H_wq = torch.matmul(J_wq_T, J_wq)   #[B,M,2,2]
    u = - torch.matmul(J_wq_T, r_depth).squeeze(-1) #[B,M,2]
    ### 2: construct linear system ###

    Jz = Jz.reshape(B, N, ht*wd, -1)    #[B,N,ht*wd,2] 
    # here Jz does not contain the negative sign in the residual term 

    E_wq_d = (J_wq_T.view(B,M,2,ht*wd,-1) * J_d[:,:,None]).sum(dim=-1) #[B,M,2,ht*wd]

    w = w.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    r = r.view(B, N, ht*wd, -1) #[B,N,ht*wd,2]
    wk = torch.sum(-w*r*Jz, dim=-1) #[B,N,ht*wd]
    Ck = torch.sum(w*(-Jz)*(-Jz), dim=-1) #[B,N,ht*wd]

    # only optimize keyframe poses
    P = torch.div(P,rig,rounding_mode="trunc")-fixedp
    ii = torch.div(ii,rig,rounding_mode="trunc")-fixedp
    jj = torch.div(jj,rig,rounding_mode="trunc")-fixedp

    H_wq = safe_scatter_add_mat(H_wq,ll,ll,M,M)       #[B,M*M,2,2]
    E_wq_d = safe_scatter_add_mat(E_wq_d,ll,ll,M,M)      #[B,M*M,2,ht*wd]
    C_proj = safe_scatter_add_vec(Ck, kk, M)                #[B,M,ht*wd]
    u = safe_scatter_add_vec(u, ll, M)                      #[B,M,2]

    # C = C + eta.view(*C.shape) #+ 1e-7
    C_depth = (J_d*J_d).view(B,M,ht*wd)
    # C = C_proj + C_depth + (1-C_depth)*eta.view(*C_proj.shape)             #[B,M,ht*wd]
    C = C_proj + C_depth + eta.view(*C_proj.shape) #+ 1e-7

    w_proj = safe_scatter_add_vec(wk, kk, M)                               #[B,M,ht*wd]
    w = -w_proj - (J_d*r_depth).view(B,M,ht*wd)  #[B,M,ht*wd]
    H = H_wq.view(B, M, M, 2, 2)
    E = E_wq_d.view(B, M, M, 2, ht*wd)
    ### 3: solve the system ###    
    dwq, dz = schur_solve(H, E, C, u, w, ep, lm)
    # dwq [B,M,2]
    # dz [B,M,ht*wd]
    ### 4: apply retraction ###
    # poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    wqs = wq_retr(wqs,dwq,kx)
    # disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps , wqs






def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

