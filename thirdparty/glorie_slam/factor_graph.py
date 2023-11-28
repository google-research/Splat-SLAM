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
import numpy as np

from thirdparty.glorie_slam.modules.droid_net import CorrBlock, AltCorrBlock
import thirdparty.glorie_slam.geom.projective_ops as pops
from copy import deepcopy


class FactorGraph:
    # mainly inherited from GO-SLAM
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # operator at 1/8 resolution
        self.ht = ht = video.ht // self.video.down_scale
        self.wd = wd = video.wd // self.video.down_scale

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.ii = None
        self.jj = None
        self.age = None
        self.corr = None
        self.damping = None
        self.net = None
        self.inp = None
        self.target = None 
        self.weight = None
        self.ii_inac = None
        self.jj_inac = None
        self.ii_bad = None
        self.jj_bad = None
        self.target_inac = None
        self.weight_inac = None

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)
        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """
        with self.video.get_lock():
            self.video.timestamp[ix] = self.video.timestamp[ix+1]
            self.video.images[ix] = self.video.images[ix+1]
            self.video.dirty[ix] = self.video.dirty[ix+1]
            self.video.npc_dirty[ix] = self.video.npc_dirty[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_up[ix] = self.video.disps_up[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]
            self.video.depth_scale[ix] = self.video.depth_scale[ix+1]
            self.video.depth_shift[ix] = self.video.depth_shift[ix+1]
            self.video.mono_disps[ix] = self.video.mono_disps[ix+1]
            self.video.valid_depth_mask[ix] = self.video.valid_depth_mask[ix+1]
            self.video.valid_depth_mask_small[ix] = self.video.valid_depth_mask_small[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, opt_type="pose_depth"):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            # bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                iters=itrs, lm=1e-4, ep=0.1, motion_only=motion_only,opt_type=opt_type)
        
            self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1


    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8, enable_wq=True):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                if v.sum() < 1:
                    continue
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)
                    self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP

            target = self.target
            weight = self.weight
            
            if enable_wq:
                opt_type = "pose_depth" if step%2==0 else "depth_scale"
            else:
                opt_type = "pose_depth"

            # dense bundle adjustment            
            self.video.ba(target, weight, damping, self.ii, self.jj, t0, t1, 
                iters=itrs, lm=1e-5, ep=1e-2, motion_only=False, opt_type=opt_type)


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1),indexing="ij")
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx,indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)


    def add_backend_proximity_factors(self, t_start, t_end, nms, radius, thresh, max_factors, beta, t_start_loop=None, loop=False):
        if t_start_loop is None or not loop:
            t_start_loop = t_start
        assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'

        ilen = (t_end - t_start_loop)
        jlen = (t_end - t_start)
        ix = torch.arange(t_start_loop, t_end)
        jx = torch.arange(t_start, t_end)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        rawd = deepcopy(d).reshape(ilen, jlen)
        d[ii - radius < jj] = np.inf
        d[d > thresh] = np.inf
        d = d.reshape(ilen, jlen)

        es = []
        # build edges within local window [i-rad, i]
        for i in range(t_start_loop, t_end):
            for j in range(max(i-radius-1, 0), i):  # j in [i-radius, i-1]
                es.append((i, j))
                es.append((j, i))
                di, dj = i-t_start_loop, j-t_start
                d[di, dj] = np.inf
                # d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf

        # distance from small to big
        vals, ix = torch.sort(d.reshape(-1), descending=False)
        ix = ix[vals<=thresh]
        ix = ix.tolist()

        loop_edges = 0

        n_neighboring = 1
        for k in ix:
            di, dj = k // jlen, k % jlen
            if d[di,dj].item() > thresh:
                continue

            if len(es) > max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            if loop:
                sub_es = []
                num_loop = 0
                for si in range(max(i-n_neighboring, t_start_loop), min(i+n_neighboring+1, t_end)):
                    for sj in range(max(j-n_neighboring, t_start), min(j+n_neighboring+1, t_end)):
                        if rawd[(si-t_start_loop), (sj-t_start)] <= thresh:
                            num_loop += 1
                            if si != sj and si-sj > 20:
                                sub_es += [(si, sj)]
                # if num_loop > int(((n_neighboring * 2 + 1) ** 2) * 0.5):
                es += sub_es
                loop_edges += len(sub_es)
            else:
                es += [(i, j), ]
                es += [(j, i), ]

            d[max(0, di-nms):min(ilen, di+nms+1), max(0, dj-nms):min(jlen, dj+nms+1)] = np.inf


        if len(es) < 3 or (loop and loop_edges==0):
            return 0

        ii, jj = torch.tensor(es, device=self.device).unbind(dim=-1)

        self.add_factors(ii, jj, remove=True)
        edge_num = len(self.ii)

        return edge_num