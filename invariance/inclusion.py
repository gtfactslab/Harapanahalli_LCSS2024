import numpy as np
import interval
# from interval import NotIntervalException
import sympy as sp

class InclusionFunction :
    def __init__(self, f_eqn, x_vars) -> None:
        self.f_eqn = sp.Matrix(f_eqn)
        self.x_vars = tuple(x_vars)

    def __call__ (self, x) :
        return NotImplementedError

class NaturalInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f = sp.lambdify((x_vars,), f_eqn, 'numpy')
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)
        return np.array(self.f(x)).reshape(-1)

class Ordering :
    def __init__(self, ordering) -> None:
        self.ordering = tuple(ordering)

    def __getitem__ (self, key) :
        return self.ordering[key]
    
    def __setitem__ (self, key, item) :
        self.ordering[key] = item
    
    def __repr__ (self) :
        return repr(self.ordering)
        
    def __str__ (self) :
        return str(self.ordering)
    
    def __len__(self) :
        return len(self.ordering)

class OrderingIter :
    def __init__(self, ordering) :
        self.ordering = ordering
        self.idx = 0
        self.length = len(ordering)
    
    def __iter__(self) :
        return self

    def __next__ (self) :
        if self.idx < self.length :
            ret = self.ordering[self.idx]
            self.idx += 1
            return ret
        raise StopIteration

def standard_ordering (n) :
    return [Ordering(range(n))]

def two_orderings (n) :
    return [Ordering(range(n)), Ordering(reversed(range(n)))]

# R^n --> R^m
class CenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars) -> None:
        super().__init__(f_eqn, x_vars)
        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
    
    def __call__(self, x) :
        cent, _ = get_cent_pert(x)
        return self.f(cent).reshape(-1) + self.Df_x(x)@(x - cent)

class MixedCenteredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, orderings=None) -> None:
        super().__init__(f_eqn, x_vars)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        self.Df_x_i = [sp.lambdify((x_vars,), self.Df_x_sym[:,i]) for i in range(self.n)]

        self.orderings = orderings if orderings is not None else standard_ordering(self.n)
    
    def __call__(self, x) :
        ret = []
        cent, _ = get_cent_pert(x)

        for ordering in self.orderings :
            ret_i = 0
            xmcent = (x - cent)
            x_jac = np.copy(cent).astype(np.interval)
            for i in range (self.n) :
                x_jac[ordering[i]] = x[ordering[i]]
                J_i = self.Df_x_i[i](x_jac)
                ret_i = ret_i + J_i@np.atleast_1d(xmcent[i])
            ret.append(as_lu(ret_i + self.f(cent).reshape(-1)).T.reshape(-1))

        ret = np.array(ret)
        l, u = np.max(ret[:,:self.m], axis=0), np.min(ret[:,self.m:], axis=0)
        return get_iarray(l,u)

class Corner :
    def __init__(self, edge) -> None:
        self.edge = tuple(edge)

    def __getitem__ (self, key) :
        return self.edge[key]
    
    def __setitem__ (self, key, item) :
        self.edge[key] = item
    
    def __repr__ (self) :
        return repr(self.edge)
        
    def __str__ (self) :
        return str(self.edge)

    def __len__(self) :
        return len(self.edge)
        
class CornerIter :
    def __init__(self, corner) -> None:
        self.corner = corner
        self.idx = 0
        self.length = len(corner)
    
    def __iter__(self) :
        return self
    
    def __next__ (self) :
        if self.idx < self.length :
            ret = self.corner[self.idx]
            self.idx += 1
            return ret
        raise StopIteration
        
def two_corners (n) :
    return [Corner((0,)*n), Corner((1,)*n)]

def all_corners (n) :
    return [Corner(tuple(reversed([(i//(2**j)) % 2 for j in range(n)]))) for i in range(2**n)]

# f_i should be a list
def _cornered_algorithm (corners, _J, J_, f, x, as_iarray=True) :
    _xx_ = as_lu(x).T.reshape(-1,1)
    _Jp, _Jn = d_positive(_J)
    J_p, J_n = d_positive(J_)
    m, n = _J.shape

    B = np.empty((2*m, 2*n))
    b = np.empty((2*m, 1))

    ret = []

    for corner in corners :
        replaced = np.array([(x[i].l if corner[i] == 0 else x[i].u) for i in range(n)])
        for i in range(n) :
            if corner[i] == 0 :
                B[:m,i]   = -_Jn[:,i]
                B[:m,i+n] =  _Jn[:,i]
                B[m:,i]   = -J_p[:,i]
                B[m:,i+n] =  J_p[:,i]
            else :
                B[:m,i]   =  J_p[:,i]
                B[:m,i+n] = -J_p[:,i]
                B[m:,i]   =  _Jn[:,i]
                B[m:,i+n] = -_Jn[:,i]

        # print(f(replaced))
        b[:m,0] = f(replaced).reshape(-1)
        b[m:,0] = b[:m,0]
        # print('B', B)
        # print('b', b)
        # print('res', (B@_xx_ + b).reshape(-1))
        ret.append((B@_xx_ + b).reshape(-1))
    
    ret = np.array(ret)
    # print(ret)
    l, u = np.max(ret[:,:m], axis=0), np.min(ret[:,m:], axis=0)
    if as_iarray :
        return get_iarray(l,u)
    else :
        return (l, u)

# J_i should be a list of n functions.
def _mixed_cornered_algorithm (orderings, corners, J_i, f, x, m, n, as_iarray=True) : 
    _xx_ = as_lu(x).T.reshape(-1,1)
    B = np.empty((2*m, 2*n))
    b = np.empty((2*m, 1))

    ret = []

    for corner in corners :
        for ordering in orderings :
            replaced = np.array([(x[i].l if corner[i] == 0 else x[i].u) for i in range(n)])
            x_jac = np.copy(replaced).astype(np.interval)
            for i in range (n) :
                x_jac[ordering[i]] = x[ordering[i]]
                # # print('x_jac',x_jac)
                _J, J_ = get_lu(J_i[i](x_jac))
                # # print(f'J{i}', J_i[i](x_jac))
                _Jp, _Jn = d_positive(_J.reshape(-1))
                J_p, J_n = d_positive(J_.reshape(-1))
                if corner[i] == 0 :
                    B[:m,i]   = -_Jn
                    B[:m,i+n] =  _Jn
                    B[m:,i]   = -J_p
                    B[m:,i+n] =  J_p
                else :
                    B[:m,i]   =  J_p
                    B[:m,i+n] = -J_p
                    B[m:,i]   =  _Jn
                    B[m:,i+n] = -_Jn
            b[:m,0] = f(replaced).reshape(-1)
            b[m:,0] = b[:m,0]
            # # print('B', B)
            # # print('b', b)
            # # print('res', (B@_xx_ + b).reshape(-1))
            # # print()
            ret.append((B@_xx_ + b).reshape(-1))

    ret = np.array(ret)
    # print(ret)
    l, u = np.max(ret[:,:m], axis=0), np.min(ret[:,m:], axis=0)
    if as_iarray :
        return get_iarray(l,u)
    else :
        return (l, u)

# R^n --> R^m
class CorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, corners=None) -> None:
        super().__init__(f_eqn, x_vars)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        # self.f = None
        # self.Df_x = None

        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.corners = corners if corners is not None else two_corners(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        J = self.Df_x(x)
        # # print(J)
        _J, J_ = get_lu(J)

        return _cornered_algorithm(self.corners, _J, J_, self.f, x)

        
# R^n --> R^m
class MixedCorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, orderings=None, corners=None) -> None:
        super().__init__(f_eqn, x_vars)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        # # print([self.Df_x_sym[:,i] for i in range(self.n)])
        self.Df_x_i = [sp.lambdify((x_vars,), self.Df_x_sym[:,i]) for i in range(self.n)]
        # self.f = None
        # self.Df_x = None

        self.orderings = orderings if orderings is not None else standard_ordering(self.n)
        self.corners = corners if corners is not None else two_corners(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        # J_i = [lambda x : get_lu(Df_x_col(x)) for Df_x_col in self.Df_x_i]

        return _mixed_cornered_algorithm(self.orderings, self.corners, self.Df_x_i, self.f, x, self.m, self.n)
        

class FastCorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, corners=None) -> None:
        super().__init__(f_eqn, x_vars)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        # self.f = None
        # self.Df_x = None

        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.corners = corners if corners is not None else two_corners(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        J = self.Df_x(x)
        ret = []
        for corner in self.corners :
            replaced = np.array([(x[i].l if corner[i] == 0 else x[i].u) for i in range(self.n)])
            ret.append(as_lu(self.f(replaced).reshape(-1) + J@(x - replaced)).T.reshape(-1))

        ret = np.array(ret)
        l, u = np.max(ret[:,:self.m], axis=0), np.min(ret[:,self.m:], axis=0)
        return get_iarray(l,u)

class FastMixedCorneredInclusionFunction (InclusionFunction) :
    def __init__(self, f_eqn, x_vars, orderings=None, corners=None) -> None:
        super().__init__(f_eqn, x_vars)
        self.n = len(x_vars)
        self.m = len(f_eqn)

        self.f_lam = sp.lambdify((x_vars,), f_eqn, 'numpy')
        self.f = lambda x : np.array(self.f_lam(x))
        self.f_i = [sp.lambdify((x_vars,), f_eqn_i, 'numpy') for f_eqn_i in self.f_eqn]
        self.Df_x_sym = self.f_eqn.jacobian((x_vars,))
        self.Df_x = sp.lambdify((x_vars,), self.Df_x_sym)
        # # print([self.Df_x_sym[:,i] for i in range(self.n)])
        self.Df_x_i = [sp.lambdify((x_vars,), self.Df_x_sym[:,i]) for i in range(self.n)]
        # self.f = None
        # self.Df_x = None

        self.orderings = orderings if orderings is not None else standard_ordering(self.n)
        self.corners = corners if corners is not None else two_corners(self.n)
    
    def __call__(self, x) :
        # if x.dtype != np.interval :
        #     raise NotIntervalException(x)

        # J_i = [lambda x : get_lu(Df_x_col(x)) for Df_x_col in self.Df_x_i]

        # return _mixed_cornered_algorithm(self.orderings, self.corners, self.Df_x_i, self.f, x, self.m, self.n)

        ret = []
        for corner in self.corners :
            for ordering in self.orderings :
                ret_i = 0
                replaced = np.array([(x[i].l if corner[i] == 0 else x[i].u) for i in range(self.n)])
                xmreplaced = (x - replaced)
                x_jac = np.copy(replaced).astype(np.interval)
                for i in range (self.n) :
                    x_jac[ordering[i]] = x[ordering[i]]
                    J_i = self.Df_x_i[i](x_jac)
                    ret_i = ret_i + J_i@np.atleast_1d(xmreplaced[i])
                ret.append(as_lu(ret_i + self.f(replaced).reshape(-1)).T.reshape(-1))
        
        ret = np.array(ret)
        l, u = np.max(ret[:,:self.m], axis=0), np.min(ret[:,self.m:], axis=0)
        return get_iarray(l,u)