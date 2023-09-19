import numpy as np
import interval
import sympy as sp
from interval import get_lu, get_iarray
from invariance.utils import d_positive

class Control :
    def __init__(self, u_len, mode='hybrid', g_tuple=None, g_eqn=None) :
        self.u_len = u_len
        # global, hybrid, or local
        self.mode = mode
        # calculation buffers
        self.uCALC = None
        self.iuCALC = None
        self.iuCALC_x = None
        self.iuCALCx_ = None
        # output feedback
        if g_eqn is not None :
            # def my_cse(exprs, symbols=None, optimizations=None, postprocess=None,
            #     order='canonical', ignore=(), list=True) :
            #     return sp.cse(exprs=exprs, symbols=sp.numbered_symbols('_dum'), optimizations='basic', 
            #                 postprocess=postprocess, order=order, ignore=ignore, list=list)
            self.g_eqn = g_eqn
            self.g_lam = sp.lambdify(g_tuple, self.g_eqn, 'numpy')
            self.g = lambda x : np.asarray(self.g_lam(x)).reshape(-1)
        else :
            self.g = lambda x : x

    def u (self, t, x) :
        pass
    
    def prime (self, x) :
        pass

    def step (self, t, x) :
        # x = self.g(x)
        # Inclusion Function
        if x.dtype == np.interval :
            self.iuCALC = self.u (t,x)
            if self.mode == 'global' :
                return self.iuCALC
            elif self.mode == 'hybrid' or self.mode == 'local' :
                d = len(x)
                if self.iuCALC_x is None :
                    self.iuCALC_x = np.empty((d, self.u_len), dtype=np.interval)
                    self.iuCALCx_ = np.empty((d, self.u_len), dtype=np.interval)
                for i in range(d) :
                    xi = np.copy(x); 

                    tmpi = x[i]; tmpi.u = x[i].l; xi[i] = tmpi
                    self.prime(xi) if self.mode == 'local' else None
                    self.iuCALC_x [i,:] = self.u (t, xi)

                    tmpi = x[i]; tmpi.l = x[i].u; xi[i] = tmpi
                    self.prime(xi) if self.mode == 'local' else None
                    self.iuCALCx_ [i,:] = self.u (t, xi)
                return self.iuCALC_x, self.iuCALCx_

        # Regular PW Control
        else :
            # x = self.g(x)
            self.uCALC = self.u (t, x)
            return self.uCALC


    def __call__(self, t, x) : 
        x = self.g(x)
        return self.step(t,x)

class LinearControl (Control) :
    def __init__(self, K, mode='hybrid'):
        super().__init__(K.shape[0], mode)
        self.K = K

    def u (self, t, x) :
        return (self.K @ x).reshape(-1)

# class AffineControl (Control) :
#     def __init__(self, C, d, _C=None, _d=None, C_=None, d_=None, u_clip=np.interval(-np.inf, np.inf), mode='hybrid'):
#         super().__init__(C.shape[0], mode)
#         self.C = C
#         self.d = d
#         self._C = _C
#         self._d = _d
#         self.C_ = C_
#         self.d_ = d_

#     @classmethod
#     def from_perturbed (cls, _C, C_, _d, d_) :
#         return cls(None, None, _C, C_, _d, d_)
    
#     def u (self, t, x) :
#         if x.dtype == np.interval and self._C is not None :
#             _x, x_ = get_lu(x)
#             _u = self._C
#         else :
#             return self.C @ x + self.d

class NoControl (Control) :
    def __init__(self, u_len=1):
        super().__init__(u_len, 'global')
        self.uZERO = np.zeros(self.u_len)

    def u (self, t, x) :
        return self.uZERO.astype(x.dtype)

class Disturbance :
    def __init__(self, w_len) :
        self.w_len = w_len

    def w  (self, t, x) :
        pass
    
class NoDisturbance (Disturbance) :
    def __init__(self, w_len=1):
        super().__init__(w_len)
        self.wZERO = np.zeros((self.w_len))

    def w  (self, t, x) :
        return self.wZERO.astype(x.dtype)

class ConstantDisturbance (Disturbance) :
    def __init__(self, wCONST, iwCONST):
        wCONST = np.asarray(wCONST)
        iwCONST = np.asarray(iwCONST)
        super().__init__(len(wCONST))
        self.wCONST  = wCONST
        self.iwCONST = iwCONST

    def w (self, t, x) :
        if x.dtype == np.interval : 
            return self.iwCONST
        else :
            return self.wCONST

    # def cut_all (self) :
    #     partitions = []
    #     part_avg = (self._wCONST + self.w_CONST) / 2
    #     w_wh = np.concatenate((self._wCONST, self.w_CONST))

    #     for part_i in range(2**self.w_len) :
    #         part = np.copy(w_wh)
    #         for ind in range (self.w_len) :
    #             part[ind + self.w_len*((part_i >> ind) % 2)] = part_avg[ind]
    #         partitions.append(ConstantDisturbance(part[:self.w_len],part[self.w_len:],self.mode))

    #     return partitions

class UniformDisturbance (Disturbance) :
    def __init__(self, w_bound):
        self.w_bound = np.asarray(w_bound)
        super().__init__(len(self.w_bound))

    def w (self, t, x) :
        if x.dtype == np.interval :
            return self.w_bound
        else :
            return np.array([np.random.uniform(wi.l, wi.u) for wi in self.w_bound])
    
