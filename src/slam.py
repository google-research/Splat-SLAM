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
import torch
import numpy as np
from collections import OrderedDict
import torch.multiprocessing as mp
from evo.core.trajectory import PosePath3D
from evo.core import lie_algebra as lie
from thirdparty.glorie_slam.modules.droid_net import DroidNet
from thirdparty.glorie_slam.depth_video import DepthVideo
from thirdparty.glorie_slam.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed,update_cam
from src.utils.Printer import Printer,FontColor
from src.utils.eval_traj import kf_traj_eval,full_traj_eval
from src.utils.eval_utils import eval_rendering
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from thirdparty.glorie_slam.backend import Backend

class SLAM:
    def __init__(self, cfg, stream:BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.verbose:bool = cfg['verbose']
        self.only_tracking:bool = cfg['only_tracking']
        self.logger = None
        self.save_dir = cfg["data"]["output"] + '/' + cfg['scene']

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, \
        self.fx, self.fy, \
        self.cx, self.cy = update_cam(cfg)

        self.droid_net:DroidNet = DroidNet()

        self.printer = Printer(len(stream))    # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        self.video = DepthVideo(cfg,self.printer)
        self.ba = Backend(self.droid_net,self.video,self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net=self.droid_net, video=self.video,
                                                printer=self.printer, device=self.device)
        
        self.tracker:Tracker = None
        self.mapper:Mapper = None
        self.stream = stream

    def load_pretrained(self, cfg):
        droid_pretrained = cfg['tracking']['pretrained']
        state_dict = OrderedDict([
            (k.replace('module.', ''), v) for (k, v) in torch.load(droid_pretrained).items()
        ])
        state_dict['update.weight.2.weight'] = state_dict['update.weight.2.weight'][:2]
        state_dict['update.weight.2.bias'] = state_dict['update.weight.2.bias'][:2]
        state_dict['update.delta.2.weight'] = state_dict['update.delta.2.weight'][:2]
        state_dict['update.delta.2.bias'] = state_dict['update.delta.2.bias'][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(f'Load droid pretrained checkpoint from {droid_pretrained}!',FontColor.INFO)

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print('Tracking Triggered!',FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f'{self.save_dir}/mono_priors/depths', exist_ok=True)

        while(self.all_trigered < self.num_running_thread):
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print('Tracking Done!',FontColor.TRACKER)
        if self.only_tracking:
            self.terminate()
    
    def mapping(self, pipe):
        if self.only_tracking:
            self.all_trigered += 1
            return
        self.mapper =  Mapper(self, pipe)
        self.printer.print('Mapping Triggered!',FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])
        
        while(self.all_trigered < self.num_running_thread):
            pass
        self.mapper.run()
        self.printer.print('Mapping Done!',FontColor.MAPPER)

        self.terminate()
        

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)
        self.ba = Backend(self.droid_net,self.video,self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!",FontColor.TRACKER)


    def terminate(self):
        """ fill poses for non-keyframe images and evaluate """
        
        if self.cfg['tracking']['backend']['final_ba'] and self.cfg['mapping']['eval_before_final_ba']:
            self.video.save_video(f"{self.save_dir}/video.npz")
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",self.stream,self.logger,self.printer)
            except Exception as e:
                self.printer.print(e,FontColor.ERROR)

            if not self.only_tracking: 
                # prepare aligned camera list of mapped frames
                traj_est_aligned = []
                cams = self.mapper.cameras
                for kf_idx in self.mapper.video_idxs:
                    traj_est_aligned.append(np.linalg.inv(gen_pose_matrix(cams[kf_idx].R, cams[kf_idx].T)))

                traj_est_aligned = PosePath3D(poses_se3=traj_est_aligned)
                traj_est_aligned.scale(global_scale)
                traj_est_aligned.transform(lie.se3(r_a, t_a))
                rendering_result = eval_rendering(
                    self.mapper,
                    self.save_dir,
                    iteration="before_refine",
                    monocular=True,
                    mesh=self.cfg["meshing"]["mesh_before_final_ba"],
                    traj_est_aligned=list(traj_est_aligned.poses_se3),
                    global_scale=global_scale,
                    scene=self.cfg['scene'],
                    eval_mesh=True if self.cfg['dataset'] == 'replica' else False,
                    gt_mesh_path=self.cfg['meshing']['gt_mesh_path']
                )

        if self.cfg['tracking']['backend']['final_ba']:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        try:
            ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                f"{self.save_dir}/video.npz",
                f"{self.save_dir}/traj",
                "kf_traj",self.stream,self.logger,self.printer)
        except Exception as e:
            self.printer.print(e,FontColor.ERROR)

        if not self.only_tracking:
            if self.cfg['tracking']['backend']['final_ba']:
                # The final refine method includes the final update of the poses and depths
                self.mapper.final_refine(iters=self.cfg["mapping"]["final_refine_iters"]) # this performs a set of optimizations with RGBD loss to correct

            # prepare aligned camera list of mapped frames
            traj_est_aligned = []
            cams = self.mapper.cameras
            for kf_idx in self.mapper.video_idxs:
                traj_est_aligned.append(np.linalg.inv(gen_pose_matrix(cams[kf_idx].R, cams[kf_idx].T)))

            traj_est_aligned = PosePath3D(poses_se3=traj_est_aligned)
            traj_est_aligned.scale(global_scale)
            traj_est_aligned.transform(lie.se3(r_a, t_a))
            # evaluate the metrics
            rendering_result = eval_rendering(
                self.mapper,
                self.save_dir,
                iteration="after_refine",
                monocular=True,
                mesh=self.cfg["meshing"]["mesh"],
                traj_est_aligned=list(traj_est_aligned.poses_se3),
                global_scale=global_scale,
                scene=self.cfg['scene'],
                eval_mesh=True if self.cfg['dataset'] == 'replica' else False,
                gt_mesh_path=self.cfg['meshing']['gt_mesh_path']
            )

        # evaluate depth error
        self.printer.print("Evaluate sensor depth error with per frame alignment",FontColor.EVAL)
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(f"{self.save_dir}/video.npz", self.stream)
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m),FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage),FontColor.EVAL)

        self.printer.print("Evaluate sensor depth error with global alignment",FontColor.EVAL)
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(f"{self.save_dir}/video.npz", self.stream, global_scale)
        self.printer.print("Depth L1: " + str(depth_l1_g),FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m_g),FontColor.EVAL)

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f'{self.save_dir}/depth_stats.txt'
        integers = {
            'depth_l1': depth_l1,
            'depth_l1_global_scale': depth_l1_g,
            'depth_l1_mask_4m': depth_l1_max_4m,
            'depth_l1_mask_4m_global_scale': depth_l1_max_4m_g,
            'Average frame coverage': coverage, # How much of each frame uses depth from droid (the rest from Omnidata)
            'traj scaling': global_scale,
            'traj rotation': r_a,
            'traj translation': t_a,
            'traj stats': ate_statistics
        }
        # Write to the file
        with open(file_path, 'w') as file:
            for label, number in integers.items():
                file.write(f'{label}: {number}\n')

        self.printer.print(f'File saved as {file_path}',FontColor.EVAL)

        full_traj_eval(self.traj_filler,
                       f"{self.save_dir}/traj",
                       "full_traj",
                       self.stream, self.logger, self.printer)

        self.printer.print("Metrics Evaluation Done!",FontColor.EVAL)

    def run(self):

        m_pipe, t_pipe = mp.Pipe()
        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,)),
        ]

        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        self.printer.terminate()


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose