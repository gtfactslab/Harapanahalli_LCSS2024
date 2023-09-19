import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import width, get_half_intervals, as_lu, as_iarray, get_lu, get_iarray
from typing import Tuple, NamedTuple, List, overload
from invariance import ContinuousTimeSpec, NoDisturbance
from invariance.neural import NeuralNetworkControl
from invariance.system import AutonomousSystem, System, NNCSystem, NeuralNetwork
from invariance.utils import sg_box, sg_boxes, draw_iarrays, jordan_canonical, d_metzler, gen_ics_iarray, get_corners
from invariance.inclusion import Ordering, standard_ordering
import shapely.geometry as sg
import shapely.ops as so
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import time
import sympy as sp
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection

class Paralleletope :
    """Transformed interval representation
    y = Tx
    calX = T^{-1}[y]
    """
    def __init__(self, Tinv:ArrayLike, y:ArrayLike) -> None:
        """Constructor for a paralleletope, linear transformation of a hyper-rectangle [y]
        y = Tx
        \calX = T^{-1}[y]

        Args:
            Tinv (ArrayLike): Inverse linear transformation
            y (ArrayLike, dtype=np.interval): Hyper-rectangle in transformed coordinates
        """
        self.Tinv = np.asarray(Tinv)
        self.T = np.linalg.inv(Tinv)
        self.y = np.asarray(y).reshape(-1)
        if self.y.dtype != np.interval :
            raise Exception(f'Trying to construct a Paralleletope with non-interval y: {y}')
        if len(self.y) != self.Tinv.shape[1] :
            raise Exception(f'Trying to construct a Paralleletope with len(y): {len(self.y)} != Tinv.shape[1]: {self.Tinv.shape[1]}')
        
        self.corners = get_corners(self.y)
        
    def plot_2d_projection (self, ax, xi=0, yi=1, resize=False, **kwargs) :
        # Taken from scipy.spatial.convex_hull_plot_2d
        conv_hull = ConvexHull([(self.Tinv@corner)[(xi,yi),] for corner in self.corners])
        line_segments = [conv_hull.points[simplex] for simplex in conv_hull.simplices]
        ax.add_collection(LineCollection(line_segments, colors=kwargs.get('color', 'black'), linestyle='solid', linewidths=kwargs.get('linewidth', 1)))
        if resize :
            margin = 0.1 * conv_hull.points.ptp(axis=0)
            xy_min = conv_hull.points.min(axis=0) - margin
            xy_max = conv_hull.points.max(axis=0) + margin
            ax.set_xlim(xy_min[0], xy_max[0])
            ax.set_ylim(xy_min[1], xy_max[1])
    
    def gen_ics (self, N) :
        Y = gen_ics_iarray(self.y, N)
        return np.array([self.Tinv@y for y in Y])
    
    def get_corners (self) :
        return np.array([(self.Tinv@corner) for corner in self.corners])

    def __contains__ (self, x:ArrayLike) :
        """Check to see if T^{-1}[y] contains x.

        Args:
            x (ArrayLike): The vector to check
        """
        return np.all(np.subseteq(self.T@np.asarray(x, dtype=np.interval), self.y))

class EulerSetting (NamedTuple) :
    dt:float
    iters:float

class InvariantSetLocator :
    class Opts (NamedTuple) :
        initial_pert:ArrayLike = np.interval(-0.1,0.1)
        linearization_pert:ArrayLike = np.zeros(0)
        T_pert:ArrayLike = np.interval(-0.03,0.03)
        # x_eq:ArrayLike = np.zeros(0)
        t_eq:int = 100
        verbose:bool = False
        orderings:List[Ordering] = []
        eps:float = 1e-8
        axs:List[plt.Axes] = []
        first_order_method:str = 'CROWN+MILP'
        l2reg_N:int = 10000
        l2reg_tau:float = 0.01
        use_Arm:bool = False
        milp_verbose:bool = False
        TTinv:Tuple[ArrayLike,ArrayLike] = None
        eq:Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike] = None
        L:ArrayLike = None

    class Result (NamedTuple) :
        paralleletope:Paralleletope
        is_invariant:bool
        G:Tuple[ArrayLike, ArrayLike]
        TTinv:Tuple[ArrayLike,ArrayLike]
        eq:Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
        L:ArrayLike
        runtime:float
    
    class FamilyResult (NamedTuple) :
        paralleletopes:List[Paralleletope]
        results:List

    # class FamilyResult (NamedTuple) :
    #     paralleletopes:List[Paralleletope]

    def __init__(self, clsys:NNCSystem) -> None:
        self.clsys = clsys
    
    def compute_invariant_paralleletope_family (self, init_opts:Opts, 
                                                forward_euler:EulerSetting=EulerSetting(0.05, 100),
                                                reverse_euler:EulerSetting=EulerSetting(0.05, 100),
                                                plot_paralleletope=None,
                                                only_invariants=True) -> FamilyResult :
        opts = init_opts
        paralleletopes = []
        results = [] 
        fig_i = 0

        # Run forward time embedding system
        for i in tqdm(range(forward_euler.iters), disable=opts.verbose) :
            if opts.verbose :
                print(f'\n\n===== Computing Invariant Family (Forward) i={i+1}/{forward_euler.iters} =====\n')
            res = self.compute_invariant_paralleletope(opts)
            results.append(res)
            if res.is_invariant :
                paralleletopes.append(res.paralleletope)
                _y, y_ = get_lu(res.paralleletope.y)
                _G, G_ = res.G
                _y += forward_euler.dt*_G
                y_ += forward_euler.dt*G_
                y = get_iarray(_y, y_)
                opts = opts._replace(
                    initial_pert = res.paralleletope.Tinv@y + np.interval(-opts.eps,opts.eps) - res.eq[0],
                    T_pert = y - res.paralleletope.T@res.eq[0],
                    TTinv = res.TTinv,
                    L = res.L
                )
                if plot_paralleletope is not None :
                    color = 'tab:blue' if (i==0) else 'black'
                    linewidth = 3 if (i==0) else 1
                    plot_paralleletope(res.paralleletope, resize=(i==0), i=fig_i, color=color, linewidth=linewidth)
                    fig_i += 1
            else :
                if len(results) == 0 :
                    print('Failed on initial_pert.')
                break
        
        # Run reverse time embedding system
        res = results[0]
        _y, y_ = get_lu(res.paralleletope.y)
        _G, G_ = res.G
        _y += -reverse_euler.dt*_G
        y_ += -reverse_euler.dt*G_
        if not np.all(_y <= y_) :
            return InvariantSetLocator.FamilyResult(paralleletopes, results)

        y = get_iarray(_y, y_)
        opts = opts._replace(
            initial_pert = res.paralleletope.Tinv@y + np.interval(-opts.eps,opts.eps) - res.eq[0],
            T_pert = y - res.paralleletope.T@res.eq[0],
            TTinv = res.TTinv,
            L = res.L
        )
        for i in tqdm(range(reverse_euler.iters), disable=opts.verbose) :
            if opts.verbose :
                print(f'\n\n===== Computing Invariant Family (Reverse) i={i+1}/{reverse_euler.iters} =====\n')
            res = self.compute_invariant_paralleletope(opts)
            if not res.is_invariant and only_invariants :
                break
            results.insert(0,res)
            paralleletopes.insert(0,res.paralleletope)
            _y, y_ = get_lu(res.paralleletope.y)
            _G, G_ = res.G
            _y += -reverse_euler.dt*_G
            y_ += -reverse_euler.dt*G_
            if np.all(_y <= y_) :
                y = get_iarray(_y, y_)
                opts = opts._replace(
                    initial_pert = res.paralleletope.Tinv@y + np.interval(-opts.eps,opts.eps) - res.eq[0],
                    T_pert = y - res.paralleletope.T@res.eq[0],
                    TTinv = res.TTinv,
                    L = res.L
                )
                if plot_paralleletope is not None :
                    color = 'black' if res.is_invariant else 'tab:red'
                    plot_paralleletope(res.paralleletope, resize=True, i=fig_i, color=color)
                    fig_i += 1
            else :
                break

        return InvariantSetLocator.FamilyResult(paralleletopes, results)
    
    def compute_invariant_paralleletope (self, opts:Opts) -> Result :
        """Compute invariant set of a closed-loop system. Proposition []

        Args:
            opts (Opts): _description_

        Returns:
            Paralleletope: _description_
        """

        def print_v (s:str) :
            if opts.verbose :
                print(s)

        t_start = time.time()

        # Setup
        n = self.clsys.sys.xlen
        p = self.clsys.sys.ulen
        q = self.clsys.sys.wlen

        # dist = self.clsys.dist
        no_dist = NoDisturbance(q)

        if opts.eq is None :
            # Finding xeq if not specified
            onetraj = self.clsys.compute_trajectory(0,opts.t_eq,np.zeros(n),dist=no_dist)
            xeq = onetraj(opts.t_eq)
            ueq = self.clsys.control.u(0,xeq)
            weq = no_dist.w(0, xeq)
            feq = self.clsys.sys.f(xeq, ueq, weq)[0].reshape(-1)
        else :
            xeq, ueq, weq, feq = opts.eq
        x0 = xeq + opts.initial_pert

        print_v(f'xeq: {xeq}\nx0: {x0}\nueq: {ueq}\nfeq: {feq}')

        if opts.TTinv is None :
            print_v('==== Finding a state transformation of the system. ====')

            # Finding closed-loop linearization at xeq
            Aeq, Beq, Deq = self.clsys.sys.get_ABD(xeq, ueq, weq)
            Keq, deq = self.clsys.control.get_Cd(x0,'L2REG',opts.l2reg_N,opts.l2reg_tau,opts.milp_verbose)
            Acl = Aeq + Beq@Keq
            print_v(f'Aeq: {Aeq}\nBeq: {Beq}\nDeq: {Deq}\nKeq: {Keq}\ndeq: {deq}\nAcl: {Acl}')

            # Complex eigenvalue decomposition
            L, U = np.linalg.eig(Acl)
            L = np.real_if_close(L); U = np.real_if_close(U)
            reL = np.real(L); imL = np.imag(L)
            print_v(f'L: {L}\nU: {U}')
            
            # Convert to Jordan form (real)
            Tinv = np.empty_like(U, dtype=np.float64)
            real_idx = []; polar_tuples = []
            skip = False
            for i, l in enumerate(L) :
                v = np.real_if_close(U[:,i])
                if not skip :
                    if np.iscomplex(l) :
                        polar_tuples.append((i,i+1))
                        rel = np.real(l); iml = np.imag(l)
                        rev = np.real(v); imv = np.imag(v)
                        Tinv[:,i] = -rev; Tinv[:,i+1] = imv
                        skip = True
                    else :
                        real_idx.append(i)
                        Tinv[:,i] = v
                else :
                    skip = False

            Tinv[np.abs(Tinv) < opts.eps] = 0
            T = np.linalg.inv(Tinv); T[np.abs(T) < opts.eps] = 0
            # print(f'T: {T}\nTinv: {Tinv}')
        else :
            T, Tinv = opts.TTinv
            L = opts.L

        print_v(f'T: {T}')
        print_v(f'Tinv: {Tinv}')
        self.clsys.control.tighten_relu(x0, opts.milp_verbose)
        self.clsys.control.step(0, x0)

        # Mixed Jacobian Algorithm
        # TODO: Replace with multiple mixings? intersections?
        orderings = standard_ordering(n+p+q) if len(opts.orderings) == 0 else opts.orderings
        JxJuJws = []

        for ordering in orderings :
            _Jx = np.empty((n,n)); J_x = np.empty((n,n))
            _Ju = np.empty((n,p)); J_u = np.empty((n,p))
            _Jw = np.empty((n,q)); J_w = np.empty((n,q))
            xr = np.copy(xeq).astype(np.interval)
            ur = np.copy(ueq).astype(np.interval)
            wr = np.zeros(q).astype(np.interval)

            for j in range(len(ordering)) :
                i = ordering[j]
                if   i < n :
                    xr[i] = x0[i]
                    _J, J_ = get_lu(self.clsys.sys.Df_x_i[i](xr, ur, wr).astype(np.interval).reshape(-1))
                    _Jx[:,i] = _J
                    J_x[:,i] = J_
                elif i < n + p :
                    k = i - n
                    ur[k] = self.clsys.control.iuCALC[k]
                    _J, J_ = get_lu(self.clsys.sys.Df_u_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                    _Ju[:,k] = _J
                    J_u[:,k] = J_
                elif i < n + p + q :
                    k = i - n - p
                    # wr[k] = w[k]
                    _J, J_ = get_lu(self.clsys.sys.Df_w_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                    _Jw[:,k] = _J
                    J_w[:,k] = J_

            Jx = get_iarray(_Jx, J_x)
            Ju = get_iarray(_Ju, J_u)
            Jw = get_iarray(_Jw, J_w)
            JxJuJws.append((Jx, Ju, Jw))
        
        # A = (Jx + Ju@Keq)

        print_v('\n===== Constructing transformed system and Transformed Affine-Offset =====')
        # y = T(x - xeq)

        # Add Tinv to neural network
        net = self.clsys.nn
        net.seq.insert(0, torch.nn.Linear(*Tinv.shape))
        net[0].weight = torch.nn.Parameter(torch.tensor(Tinv.astype(np.float32)))
        net[0].bias = torch.nn.Parameter(torch.zeros(n,dtype=torch.float32))
        control = NeuralNetworkControl(net)

        yeq = T@xeq
        y0 = yeq + opts.T_pert
        # obox = Tinv@y0
        # subseteq = np.subseteq(obox, x0)
        Tinv_corners = [Tinv@corner for corner in get_corners(y0)]
        subseteq = [np.subseteq(Tinv_corner.astype(np.interval), x0) for Tinv_corner in Tinv_corners]
        print_v(f'Tinv@y0: {Tinv@y0}')
        print_v(f'x0: {x0}')
        
        _G, G_ = np.empty_like(y0,np.float32), np.empty_like(y0,np.float32)
        _Gn, Gn_ = np.empty_like(y0,np.float32), np.empty_like(y0,np.float32)
        
        # print(f'T: {T}\nTinv: {Tinv}')
        # A = Jx + Ju@Keq
        # Fhat = Ju@deq - Jx@xeq - Ju@ueq + feq
        # print(f'A: {A}\nFhat: {Fhat}')

        # At = T@A@Tinv
        # Fhatt = T@Fhat
        # print(f'At: {At}\nFhatt: {Fhatt}')
        
        # TODO: Add w
        print_v(f"\n===== Computing _G and G_, n={len(y0)} =====")
        for i in tqdm(range(len(y0)), disable=(not opts.verbose)) :
            # Lower Edge
            yi = np.copy(y0)
            tmpi = y0[i]; tmpi.u = y0[i].l; yi[i] = tmpi
            C, d = control.get_Cd(yi,opts.first_order_method.upper(),opts.l2reg_N,opts.l2reg_tau,opts.milp_verbose)
            _G[i] = -np.inf
            w = self.clsys.dist.w(0,Tinv@yi)
            for (Jx, Ju, Jw) in JxJuJws :
                _reti = (T@(Jx + Ju@(C@T))@Tinv@yi + T@(-Jx@xeq - Ju@ueq + Ju@d + Jw@(w - weq) + feq))[i]
                _G[i] = max(_reti.l, _G[i]) #if _reti.dtype == np.interval else _reti
                # _retni = (T@(Jx + Ju@Keq)@Tinv@yi + T@(-Jx@xeq - Ju@ueq + Ju@deq + feq))[i]
                # _Gn[i] = max(_retni.l, _Gn[i]) #if _reti.dtype == np.interval else _reti

            # Upper Edge
            yi = np.copy(y0)
            tmpi = y0[i]; tmpi.l = y0[i].u; yi[i] = tmpi
            C, d = control.get_Cd(yi,opts.first_order_method.upper(),opts.l2reg_N,opts.l2reg_tau,opts.milp_verbose)
            G_[i] = np.inf
            w = self.clsys.dist.w(0,Tinv@yi)
            for (Jx, Ju, Jw) in JxJuJws :
                ret_i = (T@(Jx + Ju@(C@T))@Tinv@yi + T@(-Jx@xeq - Ju@ueq + Ju@d + Jw@(w - weq) + feq))[i]
                G_[i] = min(ret_i.u, G_[i]) #if ret_i.dtype == np.interval else ret_i
                # ret_ni = (T@(Jx + Ju@Keq)@Tinv@yi + T@(-Jx@xeq - Ju@ueq + Ju@deq + feq))[i]
                # Gn_[i] = ret_ni.u #if _reti.dtype == np.interval else _reti

        t_end = time.time()
        runtime = t_end - t_start

        print_v('\n===== Summary =====')
        print_v(f'L: {L}')
        print_v(f'y0: {y0}')
        # print(f'obox: {obox}')
        print_v(f'subseteq: {np.all(subseteq)}')
        print_v(f'_G: {_G} ({np.all(_G >= 0)})')
        print_v(f'G_: {G_} ({np.all(G_ <= 0)})')
        print_v(f'[_G, G_] >=SE 0: {np.all(_G >= 0) and np.all(G_ <= 0)}')
        print_v(f'runtime: {runtime}')
        # print(f'_Gn: {_Gn} ({np.all(_Gn >= 0)})')
        # print(f'Gn_: {Gn_} ({np.all(Gn_ <= 0)})')
        # print(f'[_Gn, Gn_] >=SE 0: {np.all(_Gn >= 0) and np.all(Gn_ <= 0)}')

        del(net.seq[0])
        is_invariant = (np.all(_G >= 0) and np.all(G_ <= 0) and np.all(subseteq))
        if is_invariant :
            print_v('Found Forward Invariant Set!')
        else :
            print_v('Not a Forward Invariant Set.')

        return InvariantSetLocator.Result(
            Paralleletope(Tinv, y0), is_invariant, (_G, G_), (T, Tinv), (xeq, ueq, weq, feq), L, runtime
        )
