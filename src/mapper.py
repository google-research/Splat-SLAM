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

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import random
from tqdm import tqdm

from colorama import Fore, Style
from multiprocessing.connection import Connection
from munch import munchify

from src.utils.datasets import get_dataset, load_mono_depth
from src.utils.common import as_intrinsics_matrix, setup_seed

from src.utils.Printer import Printer, FontColor

from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.general_utils import rotation_matrix_to_quaternion, quaternion_multiply
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from thirdparty.monogs.utils.pose_utils import update_pose
from thirdparty.monogs.utils.slam_utils import get_loss_mapping, get_median_depth
from thirdparty.monogs.utils.camera_utils import Camera


class Mapper(object):
    """
    Mapper thread.

    """
    def __init__(self, slam, pipe:Connection):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        torch.autograd.set_detect_anomaly(True)

        self.config = slam.cfg
        self.printer:Printer = slam.printer
        if self.config['only_tracking']:
            return
        self.pipe = pipe
        self.verbose = slam.verbose

        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None

        self.dtype = torch.float32
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None
      
        self.video:DepthVideo = slam.video

        model_params = munchify(self.config["mapping"]["model_params"])
        opt_params = munchify(self.config["mapping"]["opt_params"])
        pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        self.use_spherical_harmonics = self.config["mapping"]["Training"]["spherical_harmonics"]
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.cameras_extent = 6.0

        self.set_hyperparams()

        self.device = torch.device(self.config['device'])
       
        self.frame_reader = get_dataset(
            self.config, device=self.device)

        
    def set_pipe(self, pipe):
        self.pipe = pipe

    def set_hyperparams(self):
        mapping_config = self.config["mapping"]

        self.gt_camera = mapping_config["Training"]["gt_camera"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"]["gaussian_update_offset"]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config['data']['output'] + '/' + self.config['scene']

        self.move_points = self.config['mapping']['move_points']
        self.online_plotting = self.config['mapping']['online_plotting']

        

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        # This function computes the new Gaussians to be added given a new keyframe
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )


    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
    
    def update_mapping_points(self, frame_idx, w2c, w2c_old, depth, depth_old, intrinsics, method=None):
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(means, "xyz")["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
           
            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
           
            depth = depth.to(self.device)
            frame_idxs = self.gaussians.unique_kfIDs # idx which anchored the set of points
            frame_mask = (frame_idxs==frame_idx) # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(pixel_locations[:, 0], min=0, max=width - 1)
            pixel_locations[:, 1] = torch.clamp(pixel_locations[:, 1], min=0, max=height - 1)

            # Extract the depth at those pixel locations from the new depth 
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the 
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]

            rescale_scale = (1 + 1/(means_cam[:, 2])*(depth - depth_old)).unsqueeze(-1) # shift
            # account for 0 depth values - then just do rigid deformation
            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1
        
            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean*means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(global_means, "xyz")["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(transformation.expand_as(rots[frame_mask]), rots[frame_mask])
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(rots, "rotation")["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(scales, "scaling")["scaling"]


    def get_w2c_and_depth(self, video_idx, idx, mono_depth, depth_gt, print_info=False, init=False):
        est_droid_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(video_idx,self.device)
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)
        if print_info:
            print(f"valid depth number: {valid_depth_mask.sum().item()}, " 
                    f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}")
        if valid_depth_mask.sum() < 100:
            invalid = True
            print(f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).")  
        else:
            invalid = False

        est_droid_depth[~valid_depth_mask] = 0
        if not invalid:
            mono_valid_mask = mono_depth < (mono_depth.mean()*3)
            mono_depth[mono_depth > 4*mono_depth.mean()] = 0
            from scipy.ndimage import binary_erosion
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            # Add padding around the binary_image to protect the borders
            iterations = 5
            padded_binary_image = np.pad(binary_image, pad_width=iterations, mode='constant', constant_values=1)
            structure = np.ones((3, 3), dtype=int)
            # Apply binary erosion with padding
            eroded_padded_image = binary_erosion(padded_binary_image, structure=structure, iterations=iterations)
            # Remove padding after erosion
            eroded_image = eroded_padded_image[iterations:-iterations, iterations:-iterations]
            # set mono depth to zero at mask
            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(cv2.inpaint(mono_depth, (mono_depth == 0).astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_NS)).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)

            valid_mask = torch.from_numpy(eroded_image).to(self.device)*valid_depth_mask # new

            cur_wq = self.video.get_depth_scale_and_shift(video_idx, mono_depth, est_droid_depth, valid_mask)
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]

            est_droid_depth[~valid_depth_mask] = mono_depth_wq[~valid_depth_mask]

        return est_droid_depth, w2c, invalid

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        self.printer.print("Initialized map", FontColor.MAPPER)

        # online plotting
        if self.online_plotting:
            from gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            cur_idx = self.current_window[np.array(self.current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            render_pkg = render(
                                viewpoint, self.gaussians, self.pipeline_params, self.background
                            )
            (
                image,
                depth,
            ) = (
                render_pkg["render"].detach(),
                render_pkg["depth"].detach(),
            )
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                    diff_rgb=np.abs(gt - pred))

        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # compute the visibility of the gaussians
                # Only prune on the last iteration and when we have a full window
                if prune:
                    if len(current_window) == self.window_size:
                        prune_mode = self.config["mapping"]["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True # not used it seems

                ## Opacity reset
                # self.iteration_count is a global parameter. We use gaussian reset
                # every 2001 iterations meaning if we use 60 per mapping frame
                # and there are 160 keyframes in the sequence, we do resetting
                # 4 times. Using more mapping iterations leads to more resetting
                # which can prune away more gaussians.
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.printer.print("Resetting the opacity of non-visible Gaussians", FontColor.MAPPER)
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # comment for debugging
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

        # online plotting
        if self.online_plotting:
            from gaussian_splatting.utils.image_utils import psnr
            from src.utils.eval_utils import plot_rgbd_silhouette
            import cv2
            import numpy as np
            cur_idx = current_window[np.array(current_window).argmax()]
            viewpoint = self.viewpoints[cur_idx]
            render_pkg = render(
                                viewpoint, self.gaussians, self.pipeline_params, self.background
                            )
            (
                image,
                depth,
            ) = (
                render_pkg["render"].detach(),
                render_pkg["depth"].detach(),
            )
            gt_image = viewpoint.original_image
            gt_depth = viewpoint.depth 

            if viewpoint.uid != self.video_idxs[0]: # first mapping frame is reference for exposure
                image = (torch.exp(viewpoint.exposure_a.detach())) * image + viewpoint.exposure_b.detach()

            image = torch.clamp(image, 0.0, 1.0)
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            mask = gt_image > 0
            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
            diff_depth_l1 = torch.abs(depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()

            # Add plotting 2x3 grid here
            plot_dir = self.save_dir + "/online_plots"
            plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1,
                                    psnr_score.item(), depth_l1, plot_dir=plot_dir, idx=str(cur_idx),
                                    diff_rgb=np.abs(gt - pred))
        
        return gaussian_split


    def final_refine(self, prune=False, iters=26000):
        self.printer.print("Starting final refinement", FontColor.MAPPER)

        # Do final update of depths and poses
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
            _, _, depth_gtd, _ = self.frame_reader[frame_idx]
            depth_gt_numpy = depth_gtd.cpu().numpy()
            intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)
            mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
            depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, depth_gt_numpy, init=False)
            
            # Update tracking parameters
            w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
            w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
            self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
            # Update depth for viewpoint
            self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

            if keyframe_idx in self.viewpoints:
                # Update tracking parameters
                self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                # Update depth for viewpoint
                self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

            # Update mapping parameters
            if self.move_points and self.is_kf[keyframe_idx]:
                if invalid:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                else:
                    self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                    self.depth_dict[keyframe_idx] = depth_temp # not needed since it is the last deformation but keeping for clarity.


        random_viewpoint_stack = []
        frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]

        for cam_idx, viewpoint in self.viewpoints.items():
            random_viewpoint_stack.append(viewpoint)

        for _ in tqdm(range(iters)):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
           
            rand_idx = np.random.randint(0, len(random_viewpoint_stack))
            viewpoint = random_viewpoint_stack[rand_idx]
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_mapping += get_loss_mapping(
                self.config["mapping"], image, depth, viewpoint, opacity
            )
            viewspace_point_tensor_acm.append(viewspace_point_tensor)
            visibility_filter_acm.append(visibility_filter)
            radii_acm.append(radii)
            n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # optimize the exposure compensation
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
        
        self.printer.print("Final refinement done", FontColor.MAPPER)


    def initialize(self, cur_frame_idx, viewpoint):
        # self.initialized only False at beginning for monocular MonoGS
        # in the slam_frontend.py it is used in the monocular setting
        # for some minor things for bootstrapping, but it is not relevant
        # in out "with proxy depth" setting.
        self.initialized = True
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.mapped_video_idxs = []
        self.mapped_kf_idxs = []

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)


    def add_new_keyframe(self, cur_frame_idx, idx, depth=None, opacity=None):
        rgb_boundary_threshold = self.config["mapping"]["Training"]["rgb_boundary_threshold"]
        self.mapped_video_idxs.append(cur_frame_idx)
        self.mapped_kf_idxs.append(idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        # Filter out RGB pixels where the R + G + B values < 0.01
        # valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        valid_rgb = (gt_img.sum(dim=0) > -1)[None]

        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels. THIS LINE OVERWRITES THE self.viewpoints[cur_frame_idx].depth with "initial_depth"
        return initial_depth[0].cpu().numpy()

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["mapping"]["Training"]["kf_translation"]
        kf_min_translation = self.config["mapping"]["Training"]["kf_min_translation"]
        kf_overlap = self.config["mapping"]["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        # multiply by median depth in rgb-only setting to account for scale ambiguity
        dist_check = dist > kf_translation * self.median_depth 
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame


    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.  
        """
        config = self.config

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.frame_reader.fx,
            fy=self.frame_reader.fy,
            cx=self.frame_reader.cx,
            cy=self.frame_reader.cy,
            W=self.frame_reader.W_out,
            H=self.frame_reader.H_out,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
    
        num_frames = len(self.frame_reader)

        # Initialize list to keep track of Keyframes
        self.keyframe_idxs = [] # 
        self.video_idxs = [] # keyframe numbering (note first
        # keyframe for mapping is the 7th keyframe in total)
        self.is_kf = dict() # keys are video_idx and value is boolean. This prevents trying to deform frames that were never mapped.
        # this is only a problem when the last keyframe is not mapped as this would otherwise be handled by the code.
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []

        init = True

        # Define first frame pose
        _, color, _, first_frame_c2w = self.frame_reader[0]
        intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(self.device)

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = dict()
        # global camera dictionary - updated during mapping.
        self.cameras = dict()
        self.depth_dict = dict()


        while (1):
            frame_info = self.pipe.recv()
            idx = frame_info['timestamp'] # frame index
            video_idx = frame_info['video_idx'] # keyframe index
            is_finished = frame_info['end']

            if self.verbose:
                self.printer.print(f"\nMapping Frame {idx} ...", FontColor.MAPPER)
            
            if is_finished:
                print("Done with Mapping and Tracking")
                break

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx)
                print(Style.RESET_ALL)

            self.keyframe_idxs.append(idx)
            self.video_idxs.append(video_idx)


            _, color, depth_gt, c2w_gt = self.frame_reader[idx]
            mono_depth = load_mono_depth(idx, self.save_dir).to(self.device)
            color = color.to(self.device)
            c2w_gt = c2w_gt.to(self.device) 
            depth_gt_numpy = depth_gt.numpy()
            depth_gt = depth_gt.to(self.device)

            depth, w2c, invalid = self.get_w2c_and_depth(video_idx, idx, mono_depth, depth_gt_numpy, init=False)

            if invalid:
                print("WARNING: Too few valid pixels from droid depth")
                 # online glorieslam pose and depth
                data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                        "gt_pose": w2c_gt, "idx": video_idx}
                self.is_kf[video_idx] = False
                viewpoint = Camera.init_from_dataset(
                        self.frame_reader, data, projection_matrix, 
                    )
                # update the estimated pose to be the glorie pose
                viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                viewpoint.compute_grad_mask(self.config)
                # Dictionary of Camera objects at the frame index
                # self.cameras contains all cameras.
                self.cameras[video_idx] = viewpoint
                self.pipe.send("continue")
                continue # too few valid pixels from droid depth
            
            w2c_gt = torch.linalg.inv(c2w_gt)
            self.gt_w2c_all_frames.append(w2c_gt)

            # online glorieslam pose and depth
            data = {"gt_color": color.squeeze(), "glorie_depth": depth.cpu().numpy(), "glorie_pose": w2c, \
                    "gt_pose": w2c_gt, "idx": video_idx}

            viewpoint = Camera.init_from_dataset(
                    self.frame_reader, data, projection_matrix, 
                )
            # update the estimated pose to be the glorie pose
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

            viewpoint.compute_grad_mask(self.config)
            # Dictionary of Camera objects at the frame index
            # self.cameras contains all cameras.
            self.cameras[video_idx] = viewpoint

            if init:
                self.initialize(video_idx, viewpoint)

                self.printer.print("Resetting the system", FontColor.MAPPER)
                self.reset()
                self.current_window.append(video_idx)
                # Add first depth map to depth dictionary - important for the first deformation
                # of the first frame
                self.depth_dict[video_idx] = depth
                self.is_kf[video_idx] = True # we map the first keyframe (after warmup)

                self.viewpoints[video_idx] = viewpoint
                depth = self.add_new_keyframe(video_idx, idx)
                self.add_next_kf(
                    video_idx, viewpoint, depth_map=depth, init=True
                )
                self.initialize_map(video_idx, viewpoint)
                init = False
                self.pipe.send("continue")
                continue

            # check if to add current frame as keyframe and map it, otherwise, continue tracking and deform
            # the map only once we have a keyframe to be mapped.

            # we need to render from the current pose to obtain the "n_touched" variable
            # which is used during keyframe selection
            render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )

            # compute median depth which is used during keyframe selection to account for the
            # global scale ambiguity during rgb-only SLAM 
            self.median_depth = get_median_depth(render_pkg["depth"], render_pkg["opacity"])


            # keyframe selection
            last_keyframe_idx = self.current_window[0]
            
            curr_visibility = (render_pkg["n_touched"] > 0).long()
            create_kf = self.is_keyframe(
                video_idx,
                last_keyframe_idx,
                curr_visibility,
                self.occ_aware_visibility,
            )
            if len(self.current_window) < self.window_size:
                # When we have not filled up the keyframe window size
                # we rely on just the covisibility thresholding, not the 
                # translation thresholds.
                union = torch.logical_or(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                intersection = torch.logical_and(
                    curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                ).count_nonzero()
                point_ratio = intersection / union
                create_kf = (
                    point_ratio < self.config["mapping"]["Training"]["kf_overlap"]
                )
            
            if create_kf:
                self.current_window, removed = self.add_to_window(
                    video_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                    self.current_window,
                )
                self.is_kf[video_idx] = True
            else:
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue

            last_idx = self.keyframe_idxs[-1]

            for keyframe_idx, frame_idx in zip(self.video_idxs, self.keyframe_idxs):
                # need to update depth_dict even if the last idx since this is important
                # for the first deformation of the keyframe
                _, _, depth_gtd, _ = self.frame_reader[frame_idx]
                depth_gt_numpy = depth_gtd.cpu().numpy()
                mono_depth = load_mono_depth(frame_idx, self.save_dir).to(self.device)
                # depth_temp, w2c_temp = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, depth_gt_numpy, init=not init)
                depth_temp, w2c_temp, invalid = self.get_w2c_and_depth(keyframe_idx, frame_idx, mono_depth, depth_gt_numpy, init=False)

                if keyframe_idx not in self.depth_dict and self.is_kf[keyframe_idx]:
                    self.depth_dict[keyframe_idx] = depth_temp

                # No need to move the latest pose and depth
                if frame_idx != last_idx: 
                    # Update tracking parameters
                    w2c_old = torch.cat((self.cameras[keyframe_idx].R, self.cameras[keyframe_idx].T.unsqueeze(-1)), dim=1)
                    w2c_old = torch.cat((w2c_old, torch.tensor([[0, 0, 0, 1]], device="cuda")), dim=0)
                    self.cameras[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                    # Update depth for viewpoint
                    self.cameras[keyframe_idx].depth = depth_temp.cpu().numpy()

                    if keyframe_idx in self.viewpoints:
                        # Update tracking parameters
                        self.viewpoints[keyframe_idx].update_RT(w2c_temp[:3, :3], w2c_temp[:3, 3])
                        # Update depth for viewpoint
                        self.viewpoints[keyframe_idx].depth = depth_temp.cpu().numpy()

                    # Update mapping parameters
                    if self.move_points and self.is_kf[keyframe_idx]:
                        if invalid:
                            # if the frame was invalid, we don't update the depth old and just do a rigid correction for this frame
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics, method="rigid")
                        else:
                            self.update_mapping_points(keyframe_idx, w2c_temp, w2c_old, depth_temp, self.depth_dict[keyframe_idx], intrinsics)
                            self.depth_dict[keyframe_idx] = depth_temp # line does not matter since it is the last deformation anyway
 
            # Do mapping
            # self.viewpoints contains the subset of self.cameras where we did mapping
            self.viewpoints[video_idx] = viewpoint
            depth = self.add_new_keyframe(video_idx, idx)
            self.add_next_kf(video_idx, viewpoint, depth_map=depth, init=False) # set init to True for debugging

            self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

            opt_params = []
            frames_to_optimize = self.config["mapping"]["Training"]["pose_window"]
            iter_per_kf = self.mapping_itr_num

            for cam_idx in range(len(self.current_window)):
                if self.current_window[cam_idx] == 0:
                    # Do not add GT frame pose for optimization
                    continue
                viewpoint = self.viewpoints[self.current_window[cam_idx]]
                if not self.gt_camera and self.config["mapping"]["BA"]:
                    if cam_idx < frames_to_optimize:
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_rot_delta],
                                "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"]
                                * 0.5,
                                "name": "rot_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.cam_trans_delta],
                                "lr": self.config["mapping"]["Training"]["lr"][
                                    "cam_trans_delta"
                                ]
                                * 0.5,
                                "name": "trans_{}".format(viewpoint.uid),
                            }
                        )

                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)
            
            self.map(self.current_window, iters=iter_per_kf)
            self.map(self.current_window, prune=True)

            self.pipe.send("continue")
