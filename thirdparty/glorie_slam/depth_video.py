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

import numpy as np
import torch
import lietorch
import g_droid_backends as droid_backends
import thirdparty.glorie_slam.geom.ba
from torch.multiprocessing import Value

from thirdparty.glorie_slam.modules.droid_net import cvx_upsample
import thirdparty.glorie_slam.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.BA_type = cfg['tracking']['backend']['BA_type']
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=self.device, dtype=torch.uint8)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.printer = printer

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            
            mono_depth = item[4][self.down_scale//2-1::self.down_scale,
                                 self.down_scale//2-1::self.down_scale]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def dspo(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, opt_type="pose_depth"):
        """ Disparity, Scale and Pose Optimization (DSPO) layer, 
            checked the paper (and supplementary) for detailed explanation 

            opt_type: "pose_depth",  stage 1, optimize camera poses and disparity maps together, eq.16 in the paper,
                                              same as DBA
                      "depth_scale", stage 2, optimize disparity maps, scales and shifts together, eq.17 in the paper
            stage 1 and stage 2 are run alternatingly
        """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            if opt_type == "pose_depth":
                target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
                weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, False)
                self.disps.clamp_(min=1e-5)
                return True

            elif opt_type == "depth_scale":
                poses = lietorch.SE3(self.poses[None])
                disps = self.disps[None]
                scales = self.depth_scale
                shifts = self.depth_shift
                ignore_frames = 0
                self.update_valid_depth_mask(up=False)
                curr_idx = self.counter.value-1
                mono_d = self.mono_disps[:curr_idx+1]
                est_d = self.disps[:curr_idx+1]
                valid_d = self.valid_depth_mask_small[:curr_idx+1]
                scale_t, shift_t, error_t = align_scale_and_shift(mono_d, est_d, valid_d)
                avg_disps = est_d.mean(dim=[1,2])

                scales[:curr_idx+1]=scale_t
                shifts[:curr_idx+1]=shift_t

                target_t,weight_t,eta_t,ii_t,jj_t = target,weight,eta,ii,jj

                ################################################################
                if self.mono_thres:
                    # remove the edges which contains poses with bad mono depth
                    invalid_mono_mask = (error_t/avg_disps > self.mono_thres)| \
                                        (error_t.isnan())|\
                                        (scale_t < 0)|\
                                        (valid_d.sum(dim=[1,2]) < 
                                        valid_d.shape[1]*valid_d.shape[2]*0.5)
                    invalid_mono_index, = torch.where(invalid_mono_mask.clone())
                    invalid_ii_mask = (ii<0)
                    idx_in_ii = torch.unique(ii)
                    valid_eta_mask = (idx_in_ii >= 0)
                    for idx in invalid_mono_index:
                        invalid_ii_mask =  invalid_ii_mask | (ii == idx) | (jj == idx) 
                    target_t = target[:,~invalid_ii_mask]
                    weight_t = weight[:,~invalid_ii_mask]
                    ii_t = ii[~invalid_ii_mask]
                    jj_t = jj[~invalid_ii_mask]
                    idx_in_ii_t = torch.unique(ii_t)
                    valid_eta_mask = torch.tensor([idx in idx_in_ii_t for idx in idx_in_ii]).to(self.device)
                    eta_t = eta[valid_eta_mask]
                ################################################################
                success = False
                for _ in range(itrs):
                    if self.counter.value > ignore_frames and ii_t.shape[0]>0:
                        poses, disps, wqs = thirdparty.glorie_slam.geom.ba.BA_with_scale_shift(
                            target_t,weight_t,eta_t,poses,disps,
                            self.intrinsics[None],ii_t,jj_t,
                            self.mono_disps[None],
                            scales[None],shifts[None],
                            self.valid_depth_mask_small[None], ignore_frames,
                            lm,ep,alpha=0.01
                        )
                        scales = wqs[0,:,0]
                        shifts = wqs[0,:,1]
                        success = True                    

                self.depth_scale = scales
                self.depth_shift = shifts

                self.disps = disps.squeeze(0)
                self.poses = poses.vec().squeeze(0)

                self.disps.clamp_(min=1e-5)
                return success

            else:
                raise NotImplementedError

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, motion_only=False, opt_type="pose_depth"):
        if self.BA_type == "DSPO":
            success = self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,opt_type)
            if not success:
                self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"pose_depth")
        elif self.BA_type == "DBA":
            self.dspo(target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only,"pose_depth")
        else:
            raise NotImplementedError


    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            est_disp = self.disps_up[index].clone().to(device)  # [h, w]
            est_depth = 1.0 / (est_disp)
            depth_mask = self.valid_depth_mask[index].clone().to(device)
            c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)


    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()