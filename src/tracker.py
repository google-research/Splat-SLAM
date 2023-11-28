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

from thirdparty.glorie_slam.motion_filter import MotionFilter
from thirdparty.glorie_slam.frontend import Frontend 
from thirdparty.glorie_slam.backend import Backend
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.only_tracking = slam.only_tracking
        self.output = slam.save_dir

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        self.every_kf = self.cfg['mapping']['every_keyframe']
        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)
        self.online_ba = Backend(self.net,self.video, self.cfg)
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']

        self.printer:Printer = slam.printer

    def run(self, stream:BaseDataset):
        '''
        Trigger the tracking process.
        1. check whether there is enough motion between the current frame and last keyframe by motion_filter
        2. use frontend to do local bundle adjustment, to estimate camera pose and depth image, 
            also delete the current keyframe if it is too close to the previous keyframe after local BA.
        3. run online global BA periodically by backend
        4. send the estimated pose and depth to mapper, 
            and wait until the mapper finish its current mapping optimization.
        '''
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

        number_of_kf = 0
        intrinsic = stream.get_intrinsic()
        # for (timestamp, image, _, _) in tqdm(stream):
        for i in range(len(stream)):
            timestamp, image, _, _ = stream[i]
            with torch.no_grad():
                ### check there is enough motion
                self.motion_filter.track(timestamp, image, intrinsic)
                # local bundle adjustment
                self.frontend()
            curr_kf_idx = self.video.counter.value - 1
            
            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                number_of_kf += 1
                if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                    # run online global BA every {self.ba_freq} keyframes
                    self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",FontColor.TRACKER)
                    self.online_ba.dense_ba(2)
                    prev_ba_idx = curr_kf_idx
                if (not self.only_tracking) and (number_of_kf%self.every_kf==0):
                    # inform the mapper that the estimation of current pose and depth is finished
                    self.pipe.send({"is_keyframe":True, "video_idx":curr_kf_idx,
                                    "timestamp":timestamp, "end":False})
                    self.pipe.recv()

            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()

        if not self.only_tracking:
            self.pipe.send({"is_keyframe":True, "video_idx":None,
                            "timestamp":None, "end":True})

                