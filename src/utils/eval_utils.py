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

import json
import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import trimesh

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from src.utils.datasets import load_mono_depth

import traceback
from evaluate_3d_reconstruction import run_evaluation


def eval_rendering(
    mapper,
    save_dir,
    iteration="after_refine",
    monocular=False,
    mesh=False,
    traj_est_aligned=None,
    global_scale=None,
    eval_mesh=True,
    scene=None,
    gt_mesh_path=None
):  
    dataset = mapper.frame_reader
    frames = mapper.cameras
    gaussians = mapper.gaussians
    background = mapper.background
    pipe = mapper.pipeline_params
    video_idxs = mapper.video_idxs

    mkdir_p(os.path.join(save_dir, iteration))

    keyframe_idxs = mapper.keyframe_idxs
    end_idx = len(frames) - 1

    img_pred, img_gt, saved_frame_idx = [], [], []
    
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    if mesh:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


    for k, (kf_idx, video_idx) in enumerate(zip(keyframe_idxs, video_idxs)):

        saved_frame_idx.append(video_idx)
        frame = frames[video_idx]
       
        _, gt_image, gt_depth, _= dataset[kf_idx]
        gt_depth = gt_depth.cpu().numpy()
        gt_image = gt_image.squeeze().to("cuda:0")
        # retrieve mono depth
        mono_depth = load_mono_depth(kf_idx, save_dir).to("cuda:0")
        # retrieve sensor 
        sensor_depth, _, invalid = mapper.get_w2c_and_depth(video_idx, kf_idx, mono_depth, gt_depth, init=False)
        sensor_depth = sensor_depth.cpu()

        rendering_pkg = render(frame, gaussians, pipe, background)
        rendering = rendering_pkg["render"].detach()
        depth = rendering_pkg["depth"].detach()

        gt = (gt_image.squeeze().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        # include optimized exposure compensation
        if k > 0: # first mapping frame is reference for exposure
            image = (torch.exp(frame.exposure_a.detach())) * rendering + frame.exposure_b.detach()
        else:
            image = rendering
        image = torch.clamp(image, 0.0, 1.0)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0 

        gt_depth = torch.tensor(gt_depth)
        depth = depth.detach().cpu()
        

        # compute depth errors
        depth_mask = (depth > 0) * (gt_depth > 0)
        depth = global_scale*depth
        diff_depth_l1 = torch.abs(depth - gt_depth)
        diff_depth_l1_gt = diff_depth_l1 * depth_mask
        depth_l1_gt = diff_depth_l1_gt.sum() / depth_mask.sum()
        depth_l1_array.append(depth_l1_gt)

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        # Add plotting 2x3 grid here
        plot_dir = save_dir + "/plots_" + iteration
        plot_rgbd_silhouette(gt_image, gt_depth, image, depth, diff_depth_l1_gt,
                                 psnr_score.item(), depth_l1_gt, plot_dir=plot_dir, idx='video_idx_' + str(video_idx) + "_kf_idx_" + str(kf_idx),
                                 diff_rgb=np.abs(gt - pred))

        # do volumetric TSDF fusion from which the mesh will be extracted later
        if mesh:
            # mask out the pixels where the GT mesh is non-existent. Do this with the gt depth mask
            depth[gt_depth.unsqueeze(0) == 0] = 0
            depth_o3d = np.ascontiguousarray(depth.permute(1, 2, 0).numpy().astype(np.float32))
            depth_o3d = o3d.geometry.Image(depth_o3d)
            color_o3d = np.ascontiguousarray((np.clip(image.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)*255.0).astype(np.uint8))
            color_o3d = o3d.geometry.Image(color_o3d)

            w2c_o3d = np.linalg.inv(traj_est_aligned[k]) # convert from c2w to w2c
                    
            fx = frame.fx
            fy = frame.fy
            cx = frame.cx
            cy = frame.cy
            W =  depth.shape[-1]
            H = depth.shape[1]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
            # use gt pose for debugging
            # w2c_o3d = torch.linalg.inv(pose).cpu().numpy() @ dataset.w2c_first_pose
            volume.integrate(rgbd, intrinsic, w2c_o3d)

    
    if mesh:
        # Mesh the final volumetric model
        mesh_out_file = os.path.join(save_dir, iteration, "mesh.ply")
        o3d_mesh = volume.extract_triangle_mesh()
        o3d_mesh = clean_mesh(o3d_mesh)
        o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
        print('Meshing finished.')

        # evaluate the mesh
        if eval_mesh:
            try:
                pred_ply = mesh_out_file.split('/')[-1]
                last_slash_index = mesh_out_file.rindex('/')
                path_to_pred_ply = mesh_out_file[:last_slash_index]
                gt_mesh = gt_mesh_path
                result_3d = run_evaluation(pred_ply, path_to_pred_ply, "mesh",
                                        distance_thresh=0.05, full_path_to_gt_ply=gt_mesh, icp_align=True)

                print(f"3D Mesh evaluation: {result_3d}")

            except Exception as e:
                traceback.print_exception(e)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    # rendering depth l1 error
    output["mean_depthl1"] = float(np.mean(depth_l1_array)) 

    print(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_depthl1"]}', #, depth l1 sensor: {output["mean_depthl1_sensor"]}, depth l1 to sensor: {output["mean_depthl1_to_sensor"]}', 
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    # Create gif
    create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, diff_depth_l1,
                         psnr, depth_l1, plot_dir=None, idx=None, 
                         save_plot=True, diff_rgb=None, depth_max=5, opacities=None,
                         scales=None):

    os.makedirs(plot_dir, exist_ok=True)
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14/1.55
    fig_width = fig_width * aspect_ratio
    # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
    if opacities is not None or scales is not None:
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth, cmap='jet', vmin=0, vmax=depth_max)
    axs[0, 1].set_title("Input Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title("Diff RGB L1")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=diff_depth_l1.max())
    axs[1, 2].set_title("Diff Depth L1")

    if opacities is not None:
        axs[0, 3].hist(opacities, bins=50, range=(0,1))
        axs[0, 3].set_title('Histogram of Opacities')
        axs[0, 3].set_xlabel('Opacity')
        axs[0, 3].set_ylabel('Frequency')
    if scales is not None:
        axs[1, 3].hist(scales, bins=50, range=(0, scales.max()))
        axs[1, 3].set_title('Histogram of Scales')
        axs[1, 3].set_xlabel('Scale')
        axs[1, 3].set_ylabel('Frequency')
        axs[1, 3].locator_params(axis='x', nbins=6)

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')
    fig.suptitle("frame: " + str(idx), y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()


def create_gif_from_directory(directory_path, output_filename, duration=100, online=True):
    """
    Creates a GIF from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the GIF.
    :param duration: Duration of each frame in the GIF (in milliseconds).
    """

    from PIL import Image
    import re
    # Function to extract the number from the filename
    def extract_number(filename):
        # Pattern to find a number followed by '.png'
        match = re.search(r'(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        else:
            return None


    if online:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort(key=extract_number)
    else:
        # Get all PNG files in the directory
        image_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.png')]

        # Sort the files based on the number in the filename
        image_files.sort()

    # Load images
    images = [Image.open(file) for file in image_files]

    # Convert images to the same mode and size for consistency
    images = [img.convert('RGBA') for img in images]
    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.LANCZOS) for img in images]

    # Save as GIF
    resized_images[0].save(output_filename, save_all=True, append_images=resized_images[1:], optimize=False, duration=duration, loop=0)


def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh