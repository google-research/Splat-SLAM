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
from copy import deepcopy

class Backend:
    def __init__(self, net, video, cfg):
        self.video = video
        self.update_op = net.update
        self.device = cfg['device']
        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.beta = cfg['tracking']['beta']
        self.backend_thresh = cfg['tracking']['backend']['thresh']
        self.backend_radius = cfg['tracking']['backend']['radius']
        self.backend_nms = cfg['tracking']['backend']['nms']
        self.backend_normalize = cfg['tracking']['backend']['normalize']
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        
        self.backend_loop_window = cfg['tracking']['backend']['loop_window']
        self.backend_loop_thresh = cfg['tracking']['backend']['loop_thresh']
        self.backend_loop_radius = cfg['tracking']['backend']['loop_radius']
        self.backend_loop_nms = cfg['tracking']['backend']['loop_nms']

    @torch.no_grad()
    def ba(self, t_start, t_end, steps, graph, nms, radius, thresh, max_factors, t_start_loop=None, loop=False, motion_only=False, enable_wq=True):
        """ main update """
        if t_start_loop is None or not loop:
            t_start_loop = t_start
        assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'
        edge_num = graph.add_backend_proximity_factors(t_start,t_end,nms,radius,thresh,max_factors,self.beta, t_start_loop,loop)
        if edge_num == 0:
            graph.clear_edges()
            return 0
        
        graph.update_lowmem(
            t0=t_start_loop+1,  # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
            t1=t_end,
            itrs=2,
            use_inactive=False,
            steps=steps,
            enable_wq = enable_wq
        )

        graph.clear_edges()
        return edge_num

    @torch.no_grad()
    def dense_ba(self, steps=6, enable_wq=True):
        t_start = 0
        t_end = self.video.counter.value
        nms = self.backend_nms
        radius = self.backend_radius
        thresh = self.backend_thresh
        n = t_end - t_start
        max_factors = ((radius + 2) * 2) * n
        if self.backend_normalize:
            self.video.normalize()
        graph = FactorGraph(self.video, self.update_op, device=self.device, 
                            corr_impl='alt', max_factors=max_factors)
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, 
                          thresh, max_factors, motion_only=False, enable_wq=enable_wq)

        del graph
        torch.cuda.empty_cache()
        self.video.set_dirty(t_start,t_end)
        self.video.update_valid_depth_mask()
        return n, n_edges



    @torch.no_grad()
    def loop_ba(self, t_start, t_end, steps=6, motion_only=False, local_graph=None, enable_wq=True):
        ''' loop closure, add edges with high-covisiablity'''
        radius = self.backend_loop_radius
        window = self.backend_loop_window
        max_factors = 8 * window
        nms = self.backend_loop_nms
        thresh = self.backend_loop_thresh
        t_start_loop = max(0, t_end - window)

        graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt', max_factors=max_factors)
        if local_graph is not None:
            copy_attr = ['ii', 'jj', 'age', 'net', 'target', 'weight']
            for key in copy_attr:
                val = getattr(local_graph, key)
                if val is not None:
                    setattr(graph, key, deepcopy(val))

        left_factors = max_factors - len(graph.ii)
        n_edges = self.ba(t_start, t_end, steps, graph, nms, radius, thresh, 
                          left_factors, t_start_loop=t_start_loop, loop=True, 
                          motion_only=motion_only, enable_wq=enable_wq)
        del graph
        torch.cuda.empty_cache()
        return t_end - t_start_loop, n_edges

