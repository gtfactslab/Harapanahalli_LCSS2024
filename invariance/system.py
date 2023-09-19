from __future__ import annotations
from typing import NamedTuple, List
import numpy as np
import sympy as sp
import interval
from interval import get_lu, get_iarray, has_nan
from invariance.inclusion import Corner, Ordering, two_orderings, two_corners, standard_ordering
from invariance.time import *
from invariance.neural import NeuralNetwork, NeuralNetworkControl
from invariance.control import Disturbance, NoDisturbance, Control, NoControl
from invariance.utils import d_metzler, d_positive, gen_ics_iarray, set_columns_from_corner
from pprint import pformat
from inspect import getsource
import time

class Trajectory :
    def __init__(self, t_spec:TimeSpec, t0, x0, t_alloc=None) -> None:
        self.t_spec = t_spec
        self.t0 = t0
        self.tf = t0
        t_alloc = t0 + 10 if t_alloc is None else t_alloc
        self.xx = np.empty((self.t_spec.lentt(t0,t_alloc)+1,) + x0.shape, x0.dtype)

        # self._n = lambda t : np.round((t - self.t0)/self.t_spec.t_step).astype(int)
        self.set(t0,x0)

    def _n (self, t) :
        t = np.asarray(t)
        if t.ndim == 0 :
            return round((t - self.t0)/self.t_spec.t_step)
        return np.round((t - self.t0)/self.t_spec.t_step).astype(int)

    def set (self, t, x) :
        if self._n(t) > self._n(self.tf) :
            self.tf = t
        self.xx[self._n(t),:] = x

    def plot2d (self, ax, tt, xi=0, yi=1, **kwargs) :
        tt = np.atleast_1d(tt)
        xx = self(tt)
        ax.plot(xx[:,xi], xx[:,yi], **kwargs)

    def scatter2d (self, ax, tt, xi=0, yi=1, **kwargs) :
        tt = np.atleast_1d(tt)
        xx = self(tt)
        ax.scatter(xx[:,xi], xx[:,yi], **kwargs)
    
    def plot3d (self, ax, tt, xi=0, yi=1, zi=2, **kwargs) :
        tt = np.atleast_1d(tt)
        xx = self(tt)
        ax.plot(xx[:,xi], xx[:,yi], xx[:,zi], **kwargs)

    def scatter3d (self, ax, tt, xi=0, yi=1, zi=2, **kwargs) :
        tt = np.atleast_1d(tt)
        xx = self(tt)
        ax.scatter(xx[:,xi], xx[:,yi], xx[:,zi], **kwargs)

    # def to_rs (self) :
    #     partition = 

    def _call_single(self, t) :
        n = self._n(t)
        if n <= self._n(self.tf) :
            return self.xx[n]
        else :
            raise Exception(f'Trajectory not defined at {t} \\notin [{self.t0},{self.tf}]')

    def __call__(self, t) :
        t = np.asarray(t)
        if t.ndim == 0:
            return self._call_single(t)
        not_def = np.logical_or(self._n(t) > self._n(self.tf), self._n(t) < self._n(self.t0))
        if np.any(not_def) :
            raise Exception(f'Trajectory not defined at {t[not_def]} \\notin [{self.t0},{self.tf}]')
        return self.xx[self._n(t),:]

class AffineRefine :
    # Defines a relation Mz = b.
    def __init__(self, M:np.typing.ArrayLike=None, b:np.typing.ArrayLike=None) -> None:
        self.M = M
        self.b = b

    def __call__ (self, z) :
        if self.M is None :
            return z
        if z.dtype != np.interval :
            raise Exception('Call refine with an interval')
        ret = np.copy(z)
        alpha = self.b - self.M @ z
        for j in range(self.M.shape[0]) :
            for i in range(self.M.shape[1]) :
                if np.abs(self.M[j,i]) > 1e-5 and np.norm(ret[i]) > 1e-6 :
                    alphaj = self.b[j] - (self.M[j,:i]@z[:i] + self.M[j,(i+1):]@z[(i+1):])
                    zi = alphaj / self.M[j,i]
                    ret[i] = np.interval(np.median([ret[i].l, ret[i].u, zi.l]), 
                                         np.median([ret[i].l, ret[i].u, zi.u]))
        return ret
    
class System :
    def __init__(self, x_vars, u_vars, w_vars, f_eqn, t_spec:TimeSpec,
                 ref=AffineRefine(), x_clip=np.interval(-np.inf,np.inf)) -> None:
        self.x_vars = sp.Matrix(x_vars)
        self.u_vars = sp.Matrix(u_vars)
        self.w_vars = sp.Matrix(w_vars)

        self.xlen = len(x_vars)
        self.ulen = len(u_vars)
        self.wlen = len(w_vars)

        self.t_spec = t_spec
        self.x_clip = x_clip

        self.ref = ref

        if t_spec.type == 'discrete' or t_spec.type == 'continuous' :
            self.f_eqn  = sp.Matrix(f_eqn)
        elif t_spec.type == 'discretized' :
            self.f_eqn = self.x_vars + t_spec.t_step*sp.Matrix(f_eqn)
        
        def my_cse(exprs, symbols=None, optimizations=None, postprocess=None,
            order='canonical', ignore=(), list=True) :
            return sp.cse(exprs=exprs, symbols=sp.numbered_symbols('_dum'), optimizations='basic', 
                          postprocess=postprocess, order=order, ignore=ignore, list=list)

        tuple = (x_vars, u_vars, w_vars)

        self.f     = sp.lambdify(tuple, self.f_eqn, 'numpy', cse=my_cse)
        self.f_i   = [sp.lambdify(tuple, f_eqn_i, 'numpy', cse=my_cse) for f_eqn_i in self.f_eqn]
        self.f_len = len(self.f_i)


        self.Df_x_sym = self.f_eqn.jacobian(x_vars)
        self.Df_u_sym = self.f_eqn.jacobian(u_vars)
        self.Df_w_sym = self.f_eqn.jacobian(w_vars)

        # if (not self.Df_x_sym.free_symbols) and (not self.Df_u_sym.free_symbols) and (not self.Df_w_sym.free_symbols):
        if False:
            self.A = sp.lambdify((), self.Df_x_sym, 'numpy', cse=my_cse)()[0]
            self.B = sp.lambdify((), self.Df_u_sym, 'numpy', cse=my_cse)()[0]
            self.Bp, self.Bn = d_positive(self.B)
            self.D = sp.lambdify((), self.Df_w_sym, 'numpy', cse=my_cse)()[0]
            self.const = self.f(np.zeros(len(x_vars)), np.zeros(len(u_vars)), np.zeros(len(w_vars)))[0].reshape(-1)
            self.type = 'linear'
        else :
            self.Df_x = sp.lambdify(tuple, self.Df_x_sym, 'numpy', cse=my_cse)
            self.Df_u = sp.lambdify(tuple, self.Df_u_sym, 'numpy', cse=my_cse)
            self.Df_w = sp.lambdify(tuple, self.Df_w_sym, 'numpy', cse=my_cse)
            self.Df_x_i = [sp.lambdify(tuple, self.Df_x_sym[:,i], 'numpy') for i in range(self.xlen)] 
            self.Df_u_i = [sp.lambdify(tuple, self.Df_u_sym[:,i], 'numpy') for i in range(self.ulen)] 
            self.Df_w_i = [sp.lambdify(tuple, self.Df_w_sym[:,i], 'numpy') for i in range(self.wlen)] 
            self.type = 'nonlinear'


    def __str__ (self) :
        return f'''{self.type.title()} {str(self.t_spec)} System with 
            \r  {'xdot' if self.t_spec.type == 'continuous' or self.t_spec.type == 'discretized' else 'x+'} = f(x,u,w) = {str(self.f_eqn)}'''

    def get_ABD (self, x, u, w) :
        if self.type == 'linear' :
            return self.A, self.B, self.D
        else :
            return self.Df_x(x, u, w)[0].astype(x.dtype), self.Df_u(x, u, w)[0].astype(x.dtype), self.Df_w(x, u, w)[0].astype(x.dtype)

class ControlledSystem :
    def __init__(self, sys:System, control:Control, interc_mode=None,
                 dist:Disturbance=NoDisturbance(1), xclip=np.interval(-np.inf,np.inf)) :
        self.sys = sys
        self.control = control

        # Set default interconnection modes
        if interc_mode is None :
            if self.sys.t_spec.type == 'continuous' :
                interc_mode = 'hybrid'
            else :
                interc_mode = 'global'

        self.control.mode = interc_mode
        self.dist = dist
        self.xclip = xclip
    
    # Returns x_{t+1} given x_t.
    def func (self, t, x, dist=None) :
        # self.control.step(t, self.sys.g(x))
        # print(t)
        dist = self.dist if dist is None else dist
        # Monotone Inclusion
        if x.dtype == np.interval :
            if self.sys.t_spec.type == 'continuous' :
                # Interconnection mode by default
                _x, x_ = get_lu(x)
                d_x, dx_ = self.f_replace(x)
                _xtp1 = _x + self.sys.t_spec.t_step * d_x
                x_tp1 = x_ + self.sys.t_spec.t_step * dx_
                return self.sys.ref(get_iarray(_xtp1, x_tp1))
            else :
                return self.sys.ref(self.sys.f(x, self.control.iuCALC, dist.w(t,x))[0].reshape(-1))
        # Deterministic system
        else :
            if self.sys.t_spec.type == 'continuous' :
                return x + self.sys.t_spec.t_step*self.sys.f(x,self.control.uCALC,dist.w(t,x))[0].reshape(-1)
            else :
                return self.sys.f(x,self.control.uCALC,dist.w(t,x))[0].reshape(-1)


    # Abstract method to prepare for the next control interval.
    def prime (self, x) :
        # self.control.prime(x)
        pass

    def compute_trajectory (self, t0, tf, x0, dist=None) :
        xx = Trajectory(self.sys.t_spec, t0, x0, tf)

        for tt in self.sys.t_spec.tu(t0, tf) :
            self.control.prime(xx(tt[0]))
            self.control.step(0,xx(tt[0]))
            self.prime(xx(tt[0]))
            for j, t in enumerate(tt) :
                xx.set(t + self.sys.t_spec.t_step, self.func(t, xx(t), dist))

        return xx
    
    def compute_trajectory_uu (self, t0, tf, x0) :
        xx = Trajectory(self.sys.t_spec, t0, x0, tf)
        uu = []

        for tt in self.sys.t_spec.tu(t0, tf) :
            self.control.prime(xx(tt[0]))
            self.control.step(0,xx(tt[0]))
            uu.append(self.control.uCALC)
            self.prime(xx(tt[0]))
            for j, t in enumerate(tt) :
                xx.set(t + self.sys.t_spec.t_step, self.func(t, xx(t)))

        return xx, np.array(uu)
    
    def compute_mc_trajectories (self, t0, tf, x0, N) :
        if x0.dtype != np.interval :
            raise Exception('Need an interval to generate MC initial conditions.')
        
        return [self.compute_trajectory(t0, tf, mc_x0) for mc_x0 in gen_ics_iarray(x0, N)]
    
    def f_replace (self, x, iuCALC_x=None, iuCALCx_=None) :
        # ret = np.empty_like(x)
        d_x, dx_ = np.empty_like(x,np.float32), np.empty_like(x,np.float32)
        if iuCALC_x is None :
            iuCALC_x = self.control.iuCALC_x
        if iuCALCx_ is None :
            iuCALCx_ = self.control.iuCALCx_
        
        for i in range(len(x)) :
            _ui = iuCALC_x[i,:] if self.control.mode != 'global' else self.control.iuCALC
            ui_ = iuCALCx_[i,:] if self.control.mode != 'global' else self.control.iuCALC
            xi = np.copy(x)

            tmpi = x[i]; tmpi.u = x[i].l; xi[i] = tmpi
            xi = self.sys.ref(xi)
            _reti = np.interval(self.sys.f_i[i] (xi, _ui, self.dist.w(0,xi))[0])
            d_x[i] = _reti.l #if _reti.dtype == np.interval else _reti

            xi = np.copy(x)
            tmpi = x[i]; tmpi.l = x[i].u; xi[i] = tmpi
            xi = self.sys.ref(xi)
            ret_i = np.interval(self.sys.f_i[i] (xi, ui_, self.dist.w(0,xi))[0])
            dx_[i] = ret_i.u #if ret_i.dtype == np.interval else ret_i
        return d_x, dx_

class AutonomousSystem (ControlledSystem) :
    def __init__(self, x_vars, f_eqn, t_spec:TimeSpec, ref=AffineRefine()):
        u, w = sp.symbols('_0u, _0w')
        super().__init__(System(x_vars, [u], [w], f_eqn, t_spec, ref), NoControl(1), None, NoDisturbance(1))

    def __str__ (self) :
        return f'''===== Autonomous System Definition =====
            \r{self.sys.__str__()}'''

    def f_replace (self, x) :
        # ret = np.empty_like(x)
        d_x, dx_ = np.empty_like(x,np.float32), np.empty_like(x,np.float32)
        
        for i in range(len(x)) :
            xi = np.copy(x)

            tmpi = x[i]; tmpi.u = x[i].l; xi[i] = tmpi
            xi = self.sys.ref(xi)
            _reti = np.interval(self.sys.f_i[i] (xi, [0], [0])[0])
            d_x[i] = _reti.l #if _reti.dtype == np.interval else _reti

            xi = np.copy(x)
            tmpi = x[i]; tmpi.l = x[i].u; xi[i] = tmpi
            xi = self.sys.ref(xi)
            ret_i = np.interval(self.sys.f_i[i] (xi, [0], [0])[0])
            dx_[i] = ret_i.u #if ret_i.dtype == np.interval else ret_i
        return d_x, dx_


class NNCSystem (ControlledSystem) :
    class InclOpts (NamedTuple) :
        method: str = 'jacobian+interconnect'
        interc_mode: str = 'hybrid'
        orderings: List[Ordering] = list()
        corners: List[Corner] = list()

    def __init__(self, sys:System, nn:NeuralNetwork, incl_opts:InclOpts = InclOpts(), interc_mode=None,
                 dist:Disturbance=NoDisturbance(1), uclip=np.interval(-np.inf,np.inf),
                 g_tuple=None, g_eqn=None) -> None:
        self.nn = nn
        ControlledSystem.__init__(self, sys, NeuralNetworkControl(nn, incl_opts.interc_mode, uclip=uclip, g_tuple=g_tuple, g_eqn=g_eqn),
                                  interc_mode, dist)
        self.incl_opts = incl_opts
        self.e = None
        self.uj = None
        self.ujCALCx_ = None
        self.ujCALC_x = None
        self.incl_method = None
    
    def set_standard_ordering (self) :
        self.incl_opts = self.incl_opts._replace(
            orderings=standard_ordering(self.sys.xlen + self.sys.ulen + self.sys.wlen)
        )

    def set_four_corners (self) :
        _x = (-1,)*self.sys.xlen; x_ = (1,)*self.sys.xlen
        _u = (-1,)*self.sys.ulen; u_ = (1,)*self.sys.ulen
        _w = (-1,)*self.sys.wlen; w_ = (1,)*self.sys.wlen
        self.incl_opts = self.incl_opts._replace(
            corners=[Corner(_x + _u + _w), Corner(x_ + _u + w_),
                     Corner(_x + u_ + _w), Corner(x_ + u_ + w_)])

    def set_two_corners (self) :
        _x = (-1,)*self.sys.xlen; x_ = (1,)*self.sys.xlen
        _u = (-1,)*self.sys.ulen; u_ = (1,)*self.sys.ulen
        _w = (-1,)*self.sys.wlen; w_ = (1,)*self.sys.wlen
        self.incl_opts = self.incl_opts._replace(
            corners=[Corner(_x + _u + _w), Corner(x_ + u_ + w_)])
    
    def prime (self, x):
        # self.control.prime(x)
        # if self.incl_method == 'jacobian' :
        if self.sys.t_spec.type == 'continuous' :
            self.e = np.zeros(self.sys.f_len, dtype=np.interval)
            # self.control.step(0, x)
            self.uj = np.copy(self.control.iuCALC)
            self.ujCALCx_ = np.copy(self.control.iuCALCx_)
            self.ujCALC_x = np.copy(self.control.iuCALC_x)
    
    def func (self, t, x, dist=None) :
        # Returns x_{t+1} given x_t (euler disc. for continuous time based on t_spec.t_step)
        # Assumes access to pre-computed control in self.control (call control.step before this)
        # Monotone Inclusion
        if x.dtype == np.interval :
            ret = []
            for method in self.incl_opts.method.split('+') :
                self.incl_method = method
                if method == 'jacobian' :
                    if self.sys.t_spec.type == 'continuous' :
                        if self.sys.type == 'nonlinear' :
                            ret.append(np.intersection(self._nl_jac_cont(t, x), self.sys.x_clip))
                        elif self.sys.type == 'linear' :
                            ret.append(np.intersection(self._l_jac_cont(t, x), self.sys.x_clip))
                            # return self._nl_jac_cont(t, x)
                    else :
                        if self.sys.type == 'nonlinear' :
                            ret.append(np.intersection(self._nl_jac_disc(t, x), self.sys.x_clip))
                        elif self.sys.type == 'linear' :
                            ret.append(np.intersection(self._l_jac_disc(t, x), self.sys.x_clip))
                elif method == 'interconnect' :
                    if self.sys.t_spec.type == 'continuous' :
                        # Natural Inclusion with Replacements
                        _x, x_ = get_lu(x)
                        d_x, dx_ = self.f_replace(x)
                        _xtp1 = _x + self.sys.t_spec.t_step * d_x
                        x_tp1 = x_ + self.sys.t_spec.t_step * dx_
                        ret.append(np.intersection(get_iarray(_xtp1, x_tp1), self.sys.x_clip))
                    else :
                        # Natural Inclusion Function
                        ret.append(np.intersection(self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1), self.sys.x_clip))
            _ret, ret_ = get_lu(np.array(ret))
            return self.sys.ref(get_iarray(np.max(_ret, axis=0), np.min(ret_, axis=0)))
        # Deterministic system
        else :
            dist = dist if dist is not None else self.dist
            if self.sys.t_spec.type == 'continuous' :
                return x + self.sys.t_spec.t_step*self.sys.f(x,self.control.uCALC,dist.w(t,x))[0].reshape(-1)
            else :
                return self.sys.f(x,self.control.uCALC,dist.w(t,x))[0].reshape(-1)

    def F (self, t, x) :
        if x.dtype == np.interval :
            ret = []
            for method in self.incl_opts.method.split('+') :
                self.incl_method = method
                if method == 'jacobian' :
                    if self.sys.t_spec.type == 'continuous' :
                        if self.sys.type == 'nonlinear' :
                            ret.append(np.intersection(self._nl_jac_cont(t, x), self.sys.x_clip))
                        elif self.sys.type == 'linear' :
                            ret.append(np.intersection(self._l_jac_cont(t, x), self.sys.x_clip))
                            # return self._nl_jac_cont(t, x)
                    else :
                        if self.sys.type == 'nonlinear' :
                            ret.append(np.intersection(self._nl_jac_disc(t, x), self.sys.x_clip))
                        elif self.sys.type == 'linear' :
                            ret.append(np.intersection(self._l_jac_disc(t, x), self.sys.x_clip))
                elif method == 'interconnect' :
                    if self.sys.t_spec.type == 'continuous' :
                        # Natural Inclusion with Replacements
                        _x, x_ = get_lu(x)
                        d_x, dx_ = self.f_replace(x)
                        _xtp1 = _x + self.sys.t_spec.t_step * d_x
                        x_tp1 = x_ + self.sys.t_spec.t_step * dx_
                        ret.append(np.intersection(get_iarray(_xtp1, x_tp1), self.sys.x_clip))
                    else :
                        # Natural Inclusion Function
                        ret.append(np.intersection(self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0].reshape(-1), self.sys.x_clip))
            _ret, ret_ = get_lu(np.array(ret))
            return self.sys.ref(get_iarray(np.max(_ret, axis=0), np.min(ret_, axis=0)))
        # Deterministic system
        else :
            if self.sys.t_spec.type == 'continuous' :
                return x + self.sys.t_spec.t_step*self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)
            else :
                return self.sys.f(x,self.control.uCALC,self.dist.w(t,x))[0].reshape(-1)

    def __str__ (self) :
        return f'''===== Closed Loop System Definition =====
            \r{self.sys.__str__()}
            \rcontrolled by {self.control.__str__()}
            \rusing the {self.incl_method} monotone inclusion function'''

    def _nl_jac_cont (self, t, x) :
        w = self.dist.w(t, x)
        _x, x_ = get_lu(x)
        _u, u_ = get_lu(self.uj)
        _w, w_ = get_lu(w)

        n = self.sys.xlen
        p = self.sys.ulen
        q = self.sys.wlen

        _d = []; d_ = []

        _e, e_ = get_lu(self.e)

        for corner in self.incl_opts.corners :
            if self.incl_opts.orderings :
                # Mixed Cornered Algorithm
                xc = np.array([(_x[i] if corner[i] == 0 else x_[i]) for i in range(n)])
                uc = np.array([(_u[i] if corner[i+n] == 0 else u_[i]) for i in range(p)])
                wc = np.array([(_w[i] if corner[i+n+p] == 0 else w_[i]) for i in range(q)])
                fc = self.sys.f(xc, uc, wc)[0].reshape(-1)

                A, B, D = self.sys.get_ABD(x, self.uj, w)
                _A, A_ = get_lu(A); _B, B_ = get_lu(B); _D, D_ = get_lu(D)

                for ordering in self.incl_opts.orderings :
                    _Jx, J_x = set_columns_from_corner(corner[:n], _A, A_)
                    _Ju, J_u = set_columns_from_corner(corner[n:n+p], _B, B_)
                    _Jw, J_w = set_columns_from_corner(corner[n+p:], _D, D_)

                    xr = np.copy(xc).astype(np.interval)
                    ur = np.copy(uc).astype(np.interval)
                    wr = np.copy(wc).astype(np.interval)

                    for j in range(len(ordering)) :
                        i = ordering[j]
                        if   i < n :
                            xr[i] = x[i]
                            _J, J_ = get_lu(self.sys.Df_x_i[i](xr, ur, wr).astype(np.interval).reshape(-1))
                            _Jx[:,i] = _J
                            J_x[:,i] = J_
                        elif i < n + p :
                            k = i - n
                            ur[k] = self.uj[k]
                            _J, J_ = get_lu(self.sys.Df_u_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                            _Ju[:,k] = _J
                            J_u[:,k] = J_
                        elif i < n + p + q :
                            k = i - n - p
                            wr[k] = w[k]
                            _J, J_ = get_lu(self.sys.Df_w_i[k](xr, ur, wr).astype(np.interval).reshape(-1))
                            _Jw[:,k] = _J
                            J_w[:,k] = J_

                    _Bp, _Bn = d_positive(_Ju); B_p, B_n = d_positive(J_u)
                    _K = _Bp@self.control._C + _Bn@self.control.C_
                    K_ = B_p@self.control.C_ + B_n@self.control._C
                    _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

                    _H = _Jx + _K
                    H_ = J_x + K_
                    _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

                    # Bounding the difference: error dynamics for Holding effects
                    # _c = 0
                    # c_ = 0
                    _Kp, _Kn = d_positive(_K); K_p, K_n = d_positive(K_)
                    _c = - _Kn@_e - _Kp@e_
                    c_ = - K_n@e_ - K_p@_e
                    self.control.step(0, x)
                    self.e = self.e + self.sys.t_spec.t_step * self.sys.f(x, self.uj, w)[0].reshape(-1)

                    _d.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@self.control._d + _Bn@self.control.d_
                            + _Dp@_w - _Dp@w_ + fc + _c)
                    d_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@self.control._d + B_p@self.control.d_ 
                            - D_p@_w + D_p@w_ + fc + c_)
            else :
                # Cornered Algorithm
                xc = np.array([(_x[i] if corner[i] == 0 else x_[i]) for i in range(n)])
                uc = np.array([(_u[i] if corner[i+n] == 0 else u_[i]) for i in range(p)])
                wc = np.array([(_w[i] if corner[i+n+p] == 0 else w_[i]) for i in range(len(_w))])
                fc = self.sys.f(xc, uc, wc)[0].reshape(-1)

                A, B, D = self.sys.get_ABD(x, self.uj, w)
                _A, A_ = get_lu(A); _B, B_ = get_lu(B); _D, D_ = get_lu(D)

                _Jx, J_x = set_columns_from_corner(corner[:n], _A, A_)
                _Ju, J_u = set_columns_from_corner(corner[n:n+p], _B, B_)
                _Jw, J_w = set_columns_from_corner(corner[n+p:], _D, D_)

                _Bp, _Bn = d_positive(_Ju); B_p, B_n = d_positive(J_u)
                _K = _Bp@self.control._C + _Bn@self.control.C_
                K_ = B_p@self.control.C_ + B_n@self.control._C
                _Dp, _Dn = d_positive(_Jw); D_p, D_n = d_positive(J_w)

                _H = _Jx + _K
                H_ = J_x + K_
                _Hp, _Hn = d_metzler(_H); H_p, H_n = d_metzler(H_)

                # Bounding the difference: error dynamics for Holding effects
                # _c = 0
                # c_ = 0
                _Kp, _Kn = d_positive(_K); K_p, K_n = d_positive(K_)
                _c = - _Kn@_e - _Kp@e_
                c_ = - K_n@e_ - K_p@_e
                self.control.step(0, x)
                self.e = self.e + self.sys.t_spec.t_step * self.sys.f(x, self.uj, w)[0].reshape(-1)

                _d.append(_Hp@_x + _Hn@x_ - _Jx@xc - _Ju@uc + _Bp@self.control._d + _Bn@self.control.d_
                          + _Dp@_w - _Dp@w_ + fc + _c)
                d_.append(H_n@_x + H_p@x_ - J_x@xc - J_u@uc + B_n@self.control._d + B_p@self.control.d_ 
                          - D_p@_w + D_p@w_ + fc + c_)
        
        # _d = np.array(_d); d_ = np.array(d_)
        # return _d, d_

        _xtp1 = _x + self.sys.t_spec.t_step * np.max(_d, axis=0)
        x_tp1 = x_ + self.sys.t_spec.t_step * np.min(d_, axis=0)
        return get_iarray(_xtp1, x_tp1)
    
    def _nl_jac_disc (self, t, x) :
        _x, x_ = get_lu(x)
        _u, u_ = get_lu(self.control.iuCALC)
        A, B = self.sys.get_AB(x, self.control.iuCALC, self.dist.w(t,x))

        _A, A_ = get_lu(A)
        _B, B_ = get_lu(B)
        _Bp, _Bn = d_positive(_B)
        B_p, B_n = d_positive(B_)
        
        # Centered around _x
        _L = _A + _Bp@self.control._C + _Bn@self.control.C_
        L_ = A_ + B_p@self.control.C_ + B_n@self.control._C
        _Lp, _Ln = d_positive(_L)
        L_p, L_n = d_positive(L_)
        f = self.sys.f(_x,_u,self.dist.w(t,_x))[0].reshape(-1)
        d_x1 = _Lp@_x + _Ln@x_ - _A@_x - _B@_u + _Bp@self.control._d + _Bn@self.control.d_ + f
        dx_1 = L_p@x_ + L_n@_x - A_@_x - B_@_u + B_p@self.control.d_ + B_n@self.control._d + f

        # Centered around x_
        _L = A_ + B_p@self.control._C + B_n@self.control.C_
        L_ = _A + _Bp@self.control.C_ + _Bn@self.control._C
        _Lp, _Ln = d_positive(_L)
        L_p, L_n = d_positive(L_)
        f = self.sys.f(x_,u_,self.dist.w(t,x_))[0].reshape(-1)
        d_x4 = _Lp@_x + _Ln@x_ - A_@x_ - B_@u_ + B_p@self.control._d + B_n@self.control.d_ + f
        dx_4 = L_p@x_ + L_n@_x - _A@x_ - _B@u_ + _Bp@self.control.d_ + _Bn@self.control._d + f
        
        _xtp1 = np.max(np.array([d_x1,d_x4]), axis=0)
        x_tp1 = np.min(np.array([dx_1,dx_4]), axis=0)
        return get_iarray(_xtp1, x_tp1)
                    # print(self.control.iuCALC)                    # print(self.control.iuCALC)
                    # print('f: ', self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0])
                    # print('f: ', self.sys.f(x, self.control.iuCALC, self.dist.w(t,x))[0])
    def _l_jac_cont (self, t, x) :
        _x, x_ = get_lu(x)
        _L = self.sys.A + self.sys.Bp@self.control._C + self.sys.Bn@self.control.C_
        _Lp, _Ln = d_metzler(_L)
        L_ = self.sys.A + self.sys.Bp@self.control.C_ + self.sys.Bn@self.control._C
        L_p, L_n = d_metzler(L_)
        d_x = _Lp@_x + _Ln@x_ + self.sys.Bp@self.control._d + self.sys.Bn@self.control.d_ + self.sys.const
        dx_ = L_p@x_ + L_n@_x + self.sys.Bp@self.control.d_ + self.sys.Bn@self.control._d + self.sys.const

        _xtp1 = _x + self.sys.t_spec.t_step * d_x
        x_tp1 = x_ + self.sys.t_spec.t_step * dx_
        return get_iarray(_xtp1, x_tp1)

    def _l_jac_disc (self, t, x) :
        _x, x_ = get_lu(x)
        _L = self.sys.A + self.sys.Bp@self.control._C + self.sys.Bn@self.control.C_
        _Lp, _Ln = d_positive(_L)
        L_ = self.sys.A + self.sys.Bp@self.control.C_ + self.sys.Bn@self.control._C
        L_p, L_n = d_positive(L_)
        _xtp1 = _Lp@_x + _Ln@x_ + self.sys.Bp@self.control._d + self.sys.Bn@self.control.d_ + self.sys.const
        x_tp1 = L_p@x_ + L_n@_x + self.sys.Bp@self.control.d_ + self.sys.Bn@self.control._d + self.sys.const
        return get_iarray(_xtp1, x_tp1)
    
if __name__ == '__main__' :
    px, py, psi, v, u1, u2, w = sp.symbols('p_x p_y psi v u1 u2 w')
    beta = sp.atan(sp.tan(u2)/2)
    f_eqn = [
        v*sp.cos(psi + beta), 
        v*sp.sin(psi + beta), 
        v*sp.sin(beta),
        u1
    ]

    uclip = np.array([
        np.interval(-20,20),
        np.interval(-np.pi/4,np.pi/4)
    ])

    # t_spec = DiscretizedTimeSpec(0.1)
    t_spec = ContinuousTimeSpec(0.05,0.25)
    sys = System([px, py, psi, v], [u1, u2], [w], f_eqn, t_spec)
    net = NeuralNetwork('../examples/vehicle/models/100r100r2')
    clsys = NNCSystem(sys, net, 'jacobian', uclip=uclip)
    # clsys = NNCSystem(sys, net, 'interconnect', uclip=uclip)

    t_span = [0,1]
    # tt = t_spec.tt(0,1)
    # print(t_spec.tu(0,2))

    from interval import from_cent_pert
    cent = np.array([8,8,-2*np.pi/3,2])
    pert = np.array([0.1,0.1,0.01,0.01])
    # pert = np.array([0.001,0.001,0.001,0.001])
    # pert = np.array([0.01,0.01,0.01,0.01])
    x0 = from_cent_pert(cent, pert)

    from invariance.utils import run_times, draw_iarrays

    xx, times = run_times(10, clsys.compute_trajectory, t_span[0],t_span[1],x0)
    print(np.mean(times), '\pm', np.std(times))
    
    tt = t_spec.tt(t_span[0],t_span[1])
    rs = xx(tt)

    # print(xx(np.arange(0,1+0.1,0.1)))
    # print(xx(t_spec.tt(t_span[0], t_span[1])))
    # print(xx(t_spec.tt(0,1)))

    import matplotlib.pyplot as plt

    traj = clsys.compute_trajectory(t_span[0], t_span[1], cent)
    plt.plot(traj(tt)[:,0], traj(tt)[:,1])
    plt.scatter(traj(tt)[:,0], traj(tt)[:,1],s=5)
    draw_iarrays(plt, rs)
    plt.show()
