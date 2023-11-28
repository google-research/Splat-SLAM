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

import torch
from thirdparty.glorie_slam.factor_graph import FactorGraph
from thirdparty.glorie_slam.backend import Backend as LoopClosing

class Frontend:
    # mainly inherited from GO-SLAM
    def __init__(self, net, video, cfg):
        self.video = video
        self.update_op = net.update
        
        # local optimization window
        self.t1 = 0

        # frontent variables
        self.is_initialized = False

        self.max_age = cfg['tracking']['max_age']
        self.iters1 = 4*2
        self.iters2 = 2*2

        self.warmup = cfg['tracking']['warmup']
        self.beta = cfg['tracking']['beta']
        self.frontend_nms = cfg['tracking']['frontend']['nms']
        self.keyframe_thresh = cfg['tracking']['frontend']['keyframe_thresh']
        self.frontend_window = cfg['tracking']['frontend']['window']
        self.frontend_thresh = cfg['tracking']['frontend']['thresh']
        self.frontend_radius = cfg['tracking']['frontend']['radius']
        self.frontend_max_factors = cfg['tracking']['frontend']['max_factors']

        self.enable_loop = cfg['tracking']['frontend']['enable_loop']
        self.loop_closing = LoopClosing(net, video, cfg)

        self.graph = FactorGraph(
            video, net.update,
            device=cfg['device'],
            corr_impl='volume',
            max_factors=self.frontend_max_factors
        )

    def __update(self):
        """ add edges, perform update """

        self.t1 += 1
        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        for itr in range(self.iters1):
            # run DSPO for optimization
            opt_type = "pose_depth" if itr%2==0 else "depth_scale"
            self.graph.update(None, None, use_inactive=True, opt_type=opt_type)

        # set initial pose for next frame
        d = self.video.distance([self.t1-2], [self.t1-1], beta=self.beta, bidirectional=True)


        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 1)            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        else:
            cur_t = self.video.counter.value
            if self.enable_loop and cur_t > self.frontend_window:
                n_kf, n_edge = self.loop_closing.loop_ba(t_start=0, t_end=cur_t, steps=self.iters2, 
                                                         motion_only=False, local_graph=self.graph,
                                                         enable_wq=True)
                if n_edge == 0:
                    for itr in range(self.iters2):
                        opt_type = "pose_depth" if itr%2==0 else "depth_scale"
                        self.graph.update(t0=None, t1=None, use_inactive=True,opt_type=opt_type)
                self.last_loop_t = cur_t
            else:
                for itr in range(self.iters2):
                    opt_type = "pose_depth" if itr%2==0 else "depth_scale"
                    self.graph.update(t0=None, t1=None, use_inactive=True,opt_type=opt_type)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.set_dirty(self.graph.ii.min(), self.t1)
        torch.cuda.empty_cache()

    def __initialize(self):
        """ initialize the SLAM system, i.e. bootstrapping """

        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True,opt_type="pose_depth")

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True,opt_type="pose_depth")


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.timestamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.set_dirty(0, self.t1)

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            self.video.update_valid_depth_mask()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
            self.video.update_valid_depth_mask()
        
