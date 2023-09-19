import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import width, get_half_intervals, as_lu, as_iarray, get_lu, get_iarray
from typing import NamedTuple
from invariance import TimeSpec
from invariance.system import ControlledSystem
from invariance.utils import sg_box, sg_boxes, draw_iarrays
import shapely.geometry as sg
import shapely.ops as so
from tqdm import tqdm
import time

class Partition :
    _id = 0

    def __init__(self, t_spec:TimeSpec, t0, x0, 
                 depth=0, primer_depth=0, primer:bool = False) :
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        if x0.dtype != np.interval :
            raise Exception('need to initialize partition with an interval dtype')
        self.x0 = x0
        self.depth = depth
        self.primer_depth = primer_depth
        self.primer = primer

        self.xx = [x0]
        # self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.subpartitions = None

        # print(f'new partition {depth}, {primer_depth}, {primer}')

        self._id = Partition._id
        Partition._id += 1

    def _n (self, t) :
        t = np.asarray(t)
        if t.ndim == 0 :
            return round((t - self.t0)/self.t_spec.t_step)
        return np.round((t - self.t0)/self.t_spec.t_step).astype(int)

    def set (self, t, x) :
        self.xx.append(x)
        self.tf = t
        # if len(self.xx) == self._n(t) :
        #     self.xx.append(x)
        #     self.tf = t
        # else :
        #     raise Exception(f'_n(t): {self._n(t)} is not n+1: {len(self.xx)}')

    def revert (self, t) :
        # if t > self.tf :
        #     raise Exception(f'cannot revert to a future time t={t} > tf={self.tf}')
        # self.xx = self.xx[:self._n(t)+1]
        del(self.xx[self._n(t)+1:])
        self.tf = t

    def half_partition_all (self, primer:bool) :
        if self.subpartitions is None :
            primer_depth = self.primer_depth + 1 if primer else self.primer_depth
            intervals = get_half_intervals(self.xx[-1])
            self.subpartitions = [Partition(self.t_spec, self.tf, i, self.depth+1, primer_depth, primer) \
                                  for i in intervals]
        else :
            for part in self.subpartitions :
                part.half_partition_all (primer)
    
    def re_half_partition_all (self, t, x) :
        if self.subpartitions is not None :
            # print(f'repartitioning {t}')
            intervals = get_half_intervals(x)
            for part_i, part in enumerate(self.subpartitions) :
                # print(f'{part.depth}, {len(part.xx)}, {part._n(t)}, {part_i}')
                part.re_half_partition_all(t, intervals[part_i])
        else :
            self.xx[self._n(t)] = x

    def get_all (self, t) :
        if self._n(t) <= self._n(self.tf) :
            return [np.asarray(self.xx)[self._n(t)]]
        elif self.subpartitions is not None :
            boxes = []
            for part in self.subpartitions :
                boxes.extend(part.get_all(t))
            return boxes
        else :
            raise Exception(f'Partition not defined at {t} \\notin [{self.t0}, {self.tf}]')

    def draw_rs (self, ax, tt, xi=0, yi=1, **kwargs) :
        tt = np.atleast_1d(tt)
        for t in tt :
            draw_iarrays(ax, self.get_all(t), xi, yi, **kwargs)

    def area (self, t, xi=0, yi=1) :
        return so.unary_union(sg_boxes(self.get_all(t), xi, yi)).area
        # boxes = self.sg_boxes(t, xi, yi)
        # shape = so.unary_union(boxes)
        # return shape.area

    # def plot_rs_t (self, ax, tt, xi=0, color='tab:blue', **kwargs) :
    #     tivals = np.array([np.interval(tt[i],tt[i+1]) for i in range(len(tt)-1)])
    #     iarrays = np.concatenate((tivals,self(tt)[:,xi]))

    def check_safety (self, spec, tt) :
        ret = 'Y'
        for t in tt :
            # xx = self(tt)
            # print(spec(self(t)))
            s = spec(self(t))
            if s.l < 0 and s.u > 0 :
                ret = 'U'
            elif s.l < 0 and s.u < 0 :
                return 'N'
        return ret

    def check_safety_tt (self, specs, tt) :
        ret = np.empty_like(tt, dtype=str)
        # ret = np.empty_like(tt, dtype=np.interval)
        # print(specs)
        for i, t in enumerate(tt) :
            res = [spec(self(t)) for spec in specs]
            s = np.interval(np.min([r.l for r in res]), 
                            np.max([r.u for r in res]))
            if s.l > 0 and s.u > 0 :
                ret[i] = 'Y'
            if s.l < 0 and s.u > 0 :
                ret[i] = 'U'
            elif s.l < 0 and s.u < 0 :
                ret[i] = 'N'
            # ret[i] = s
        return ret

    def _call_single(self, t) :
        n = self._n(t)
        if n <= self._n(self.tf) :
            return self.xx[n]
        elif self.subpartitions is not None :
            # Shape is (subpartitions, xlen, 2 (lu))
            xx2_parts = as_lu(np.asarray([(subpart(t)) for subpart in self.subpartitions]))
            # Shape is (time, xlen)
            xx2_l = np.min(xx2_parts[:,:,0], axis=0)
            xx2_u = np.max(xx2_parts[:,:,1], axis=0)
            xx2 = get_iarray(xx2_l, xx2_u)
            return xx2

    def __call__(self,t) :
        t = np.asarray(t)
        if t.ndim == 0 :
            return self._call_single(t)

        tmask = self._n(t) <= self._n(self.tf)
        t1 = t[tmask]
        t2 = t[np.logical_not(tmask)]
        xx1 = np.asarray(self.xx)[self._n(t1)]
        if np.any(np.logical_not(tmask)) :
            if self.subpartitions is None :
                raise Exception(f'Partition not defined at {t2} \\notin [{self.t0}, {self.tf}]')
            # Shape is (subpartitions, time, xlen, 2 (lu))
            xx2_parts = as_lu(np.asarray([(subpart(t2)) for subpart in self.subpartitions]))
            # Shape is (time, xlen)
            xx2_l = np.min(xx2_parts[:,:,:,0], axis=0)
            xx2_u = np.max(xx2_parts[:,:,:,1], axis=0)
            xx2 = get_iarray(xx2_l, xx2_u)
            return np.vstack((xx1, xx2)).reshape(t.shape + (-1,))
        elif np.any(tmask) :
            return xx1.reshape(t.shape + (-1,))
        return np.array([])

class Partitioner :
    def __init__(self, clsys:ControlledSystem) -> None:
        self.clsys = clsys

    def compute_reachable_set(self, t0, tf, x0, opts) -> Partition :
        raise NotImplementedError

class UniformPartitioner (Partitioner) :
    class Opts (NamedTuple) :
        depth: int = 0
        primer_depth: int = 0
        repartition:bool = False
    
    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)

    def compute_reachable_set(self, t0, tf, x0, opts:Opts=Opts(0,0)) :
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, primer_depth=0, primer=True)
        
        for d in range(opts.depth) :
            parent.half_partition_all(d < opts.primer_depth)
        
        for tt in parent.t_spec.tu(t0, tf) :
            self.integrate_partition(parent, tt)
            if opts.repartition :
                t = tt[-1] + self.clsys.sys.t_spec.t_step
                parent.re_half_partition_all(t, parent(t))

        return parent

    def integrate_partition (self, partition:Partition, tt) :
        x0 = partition(tt[0])
        if partition.primer :
            self.clsys.control.prime(x0)
        if partition.subpartitions is None :
            self.clsys.control.step(tt[0], x0)
            self.clsys.prime(x0)
            for t in tt :
                partition.set(t + self.clsys.sys.t_spec.t_step, self.clsys.func(t, partition(t)))
        else :
            for subpartition in partition.subpartitions :
                self.integrate_partition(subpartition, tt)
    
class CGPartitioner (Partitioner) :
    class Opts (NamedTuple) :
        eps:ArrayLike
        max_depth:int = -1
        max_primer_depth:int = -1
        gamma:float = 1
        max_primer_partitions:int = -1
        max_leaf_partitions:int = -1
        max_partitions:int = -1
        enable_bar:bool = False

    def __init__(self, clsys:ControlledSystem) -> None:
        super().__init__(clsys)
        self.opts = None

    def compute_reachable_set(self, t0, tf, x0, opts:Opts) :
        self.opts = opts
        parent = Partition(self.clsys.sys.t_spec, t0, x0, 
                           depth=0, primer_depth=0, primer=True)
        
        # for d in range(opts.max_depth) :
        #     parent.half_partition_all(d < opts.max_primer_depth)
        
        for tt in tqdm(parent.t_spec.tu(t0, tf), disable=not opts.enable_bar) :
            self.integrate_partition(parent, tt)
        return parent

    def integrate_partition (self, partition:Partition, tt) :
        x0 = partition(tt[0])
        if partition.primer :
            self.clsys.control.prime(x0)
        if partition.subpartitions is None :
            self.clsys.control.step(tt[0], x0)
            self.clsys.prime(x0)
            for t in tt :
                tp1 = t + self.clsys.sys.t_spec.t_step
                partition.set(tp1, self.clsys.func(t, partition(t)))
                if partition.depth < self.opts.max_depth and \
                    partition._n(tp1) >= partition._n(t + self.opts.gamma*self.clsys.sys.t_spec.t_step) :
                    # Contraction Guided Adaptive Partitioning algorithm
                    wt0 = width(x0, self.opts.eps); mwt0 = np.max(wt0)
                    wtm = width(partition(tp1), self.opts.eps); mwtm = np.max(wtm)
                    C = mwtm / mwt0
                    if (C**(1/self.opts.gamma))*mwt0 > 1 :
                        partition.revert(tt[0])
                        sub_primer = partition.primer_depth < self.opts.max_primer_depth
                        if sub_primer and partition.primer :
                            partition.primer = False
                        partition.half_partition_all(sub_primer)
                        self.integrate_partition(partition, tt)
                        return

        else :
            for subpartition in partition.subpartitions :
                self.integrate_partition(subpartition, tt)
