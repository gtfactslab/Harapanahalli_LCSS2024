import numpy as np
import interval
from interval import get_iarray, get_lu
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset
from invariance.control import *
from invariance.utils import gen_ics_iarray
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict
import os
# from invariance.decomp import d_metzler, d_positive
import time

class NeuralNetwork (nn.Module) :
    def __init__(self, dir=None, load=True, device='cpu') :
        super().__init__()

        self.dir = dir
        self.mods = []
        self.out_len = None
        with open(os.path.join(dir, 'arch.txt')) as f :
            arch = f.read().split()

        inputlen = int(arch[0])

        for a in arch[1:] :
            if a.isdigit() :
                self.mods.append(nn.Linear(inputlen,int(a)))
                inputlen = int(a)
                self.out_len = int(a)
            else :
                if a.lower() == 'relu' :
                    self.mods.append(nn.ReLU())
                elif a.lower() == 'sigmoid' :
                    self.mods.append(nn.Sigmoid())
                elif a.lower() == 'tanh' :
                    self.mods.append(nn.Tanh())

        self.seq = nn.Sequential(*self.mods)

        if load :
            loadpath = os.path.join(dir, 'model.pt')
            self.load_state_dict(torch.load(loadpath, map_location=device))
            # print(f'Successfully loaded model from {loadpath}')

        self.device = device
        # self.dummy_input = torch.tensor([[0,0,0,0,0]], dtype=torch.float64).to(device)
        self.to(self.device)

    def forward(self, x) :
        return self.seq(x)
    
    def __getitem__(self,idx) :
        return self.seq[idx]
    
    def __str__ (self) :
        return f'neural network from {self.dir}, {str(self.seq)}'
    
    def save(self) :
        savepath = os.path.join(self.dir, 'model.pt')
        print(f'Saving model to {savepath}')
        torch.save(self.state_dict(), savepath)

class NeuralNetworkData (Dataset) :
    def __init__(self, X, U) :
        self.X = X
        self.U = U
    def __len__(self) :
        return self.X.shape[0]
    def __getitem__(self, idx) :
        return self.X[idx,:], self.U[idx,:]
    def maxU(self) :
        maxs, maxi = torch.max(self.U, axis=0)
        return maxs

class ScaledMSELoss (nn.MSELoss) :
    def __init__(self, scale, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.scale = scale
    def __call__(self, output, target) :
        return super().__call__(output/self.scale, target/self.scale)

class NeuralNetworkControl (Control) :
    def __init__(self, nn, mode='hybrid', method='CROWN', bound_opts={'relu': 'same-slope'}, device='cpu', x_len=None, u_len=None, uclip=np.interval(-np.inf,np.inf), g_eqn=None, g_tuple=None, verbose=False, custom_ops=None, **kwargs):
        super().__init__(u_len=nn.out_len if u_len is None else u_len, mode=mode, g_tuple=g_tuple, g_eqn=g_eqn)
        self.x_len = nn[0].in_features if x_len is None else x_len
        self.global_input = torch.zeros([1,self.x_len], dtype=torch.float32)
        self.nn = nn
        self.bnn = BoundedModule(nn, self.global_input, bound_opts, 
                                 device, verbose)
                                #  bound_opts={'relu': 'same-slope'})
        # self.global_input = global_input
        self.method = method
        self.device = device
        self.required_A = defaultdict(set)
        self.required_A[self.bnn.output_name[0]].add(self.bnn.input_name[0])
        self._C = None
        self.C_ = None
        self._Cp = None
        self._Cn = None
        self.C_p = None
        self.C_n = None
        self.C = None
        self._d = None
        self.d_ = None
        self.d = None
        self.uclip = uclip
        self._uclip, self.u_clip = get_lu(uclip)

        
    def u (self, t, x) :
        x = self.g(x)
        if x.dtype == np.interval :
            # Assumes .prime was called beforehand.
            # u = (self.C @ x + self.d).reshape(-1)
            _x, x_ = get_lu(x)
            _u = self._Cp @ _x + self._Cn @ x_ + self._d
            u_ = self.C_p @ x_ + self.C_n @ _x + self.d_
            # u = get_iarray(_u, u_)
            # ret_u = np.max(_u, self._uclip)
            ret_u = np.clip(_u, self._uclip, self.u_clip)
            retu_ = np.clip(u_, self._uclip, self.u_clip)
            return get_iarray(ret_u, retu_)
            # return np.intersection(u, self.uclip)
        else :
            xin = torch.tensor(x.astype(np.float32),device=self.device)
            u = self.nn(xin).cpu().detach().numpy().reshape(-1)
            return np.clip(u, self._uclip,self.u_clip)
    
    # Primes the control if to work for a range of x_xh (finds _C, C_, _d, d_)
    def prime (self, x) :
        x_pre = x
        x = self.g(x)
        if x.dtype != np.interval :
            return
            # raise Exception('Call prime with an interval array')
        _x, x_ = get_lu(x)
        x_L = torch.tensor(_x.reshape(1,-1), dtype=torch.float32)
        x_U = torch.tensor(x_.reshape(1,-1), dtype=torch.float32)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
        bt_input = BoundedTensor(self.global_input, ptb)
        # self.bnn.set_bound_opts({'optimize_bound_args': {'iteration': 0, 'lr_alpha': 0.1, }})
        self.u_lb, self.u_ub, A_dict = \
            self.bnn.compute_bounds(x=(bt_input,), method=self.method, return_A=True, 
                                    needed_A_dict=self.required_A, aux={'relu': 'same-slope'})
        self.u_lb = self.u_lb.cpu().detach().numpy()
        self.u_ub = self.u_ub.cpu().detach().numpy()

        self._C = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lA'].cpu().detach().numpy().reshape(self.u_len,-1)
        self.C_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['uA'].cpu().detach().numpy().reshape(self.u_len,-1)
        self._Cp, self._Cn = d_positive(self._C, True)
        self.C_p, self.C_n = d_positive(self.C_, True)
        self.C = get_iarray(self._C, self.C_)
        self._d = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['lbias'].cpu().detach().numpy().reshape(-1)
        self.d_ = A_dict[self.bnn.output_name[0]][self.bnn.input_name[0]]['ubias'].cpu().detach().numpy().reshape(-1)
        self.d = get_iarray(self._d, self.d_)
        # if np.any(np.abs(self._C - self.C_) > 1e-5):
        # if True:
        if False:
            print('\n_C', self._C)
            print('C_', self.C_)
            print('_d', self._d)
            print('d_', self.d_)
            # input()
        
        # self.tighten_relu(x_pre)

    def get_C (self, x, method='L2REG', l2reg_N=1000, l2reg_tau=0.0) :
        x = self.g(x)
        if x.dtype != np.interval :
            return
        if method == 'L2REG' :
            r''' Solves \min_{\alpha_i} \|u_i - X \alpha_i\|_2^2 + \tau\|\alpha_i\|_2^2
            '''
            n = self.x_len
            p = self.u_len
            reg_X = np.empty((l2reg_N,n+1))
            reg_u = np.empty((l2reg_N,p))
            np.random.seed(2023)
            reg_X0 = gen_ics_iarray(x, l2reg_N)
            Keq = np.empty((p,n))
            bias = np.empty(p)
            for i, X in enumerate(reg_X0) :
                reg_X[i,:] = np.hstack((X, 1))
                reg_u[i,:] = self.u(0, X)

            # Ridge (Tikhonov Regularized) Regression
            alpha = np.linalg.inv(reg_X.T @ reg_X + l2reg_tau*np.eye(n+1)) @ reg_X.T @ reg_u
            Keq = alpha[:-1].T
            bias = alpha[-1]
            self.setup_Cd(Keq, Keq, bias, bias)
            return Keq
        elif method == 'CROWN' :
            self.prime(x)
            return self._C

    def get_Cd (self, x, method='L2REG', l2reg_N=10000, l2reg_tau=0.01, milp_verbose=False) :
        x = self.g(x)
        if x.dtype != np.interval :
            return
        if method == 'L2REG' :
            r''' Solves \min_{\alpha_i} \|u_i - X \alpha_i\|_2^2 + \tau\|\alpha_i\|_2^2
            '''
            # print(f'HERE: {x}, {l2reg_N}, {l2reg_tau}')
            # print(self.nn)
            n = self.x_len
            p = self.u_len
            reg_X = np.empty((l2reg_N,n+1))
            reg_u = np.empty((l2reg_N,p))
            np.random.seed(2023)
            reg_X0 = gen_ics_iarray(x, l2reg_N)
            Keq = np.empty((p,n))
            bias = np.empty(p)
            for i, X in enumerate(reg_X0) :
                reg_X[i,:] = np.hstack((X, 1))
                reg_u[i,:] = self.u(0, X)

            # Ridge (Tikhonov Regularized) Regression
            alpha = np.linalg.inv(reg_X.T @ reg_X + l2reg_tau*np.eye(n+1)) @ reg_X.T @ reg_u
            Keq = alpha[:-1].T
            bias = alpha[-1]
            self.setup_Cd(Keq, Keq, bias, bias)
            self.tighten_relu(x, milp_verbose)
            self.step(0,x)
            return Keq, self.d
        elif method == 'CROWN+MILP' :
            self.prime(x)
            self.tighten_relu(x, milp_verbose)
            self.step(0,x)
            return self._C, self.d
        elif method == 'CROWN' :
            self.prime(x)
            self.step(0,x)
            return self._C, self.d

    def setup_Cd (self, _C, C_, _d, d_) :
        self._C = np.copy(_C)
        self.C_ = np.copy(C_)
        self._Cp, self._Cn = d_positive(self._C, True)
        self.C_p, self.C_n = d_positive(self.C_, True)
        self.C = get_iarray(self._C, self.C_)
        self._d = np.copy(_d)
        self.d_ = np.copy(d_)
        self.d = get_iarray(self._d, self.d_)

    def tighten_relu (self, x, verbose=False) :
        import gurobipy as gp
        from gurobipy import GRB
        import gurobi_ml as gml
        import numpy as np
        import interval

        x = self.g(x)

        if x.dtype != np.interval :
            return
            # raise Exception('Call prime with an interval array')
        _x, x_ = get_lu(x)

        gp_m = gp.Model()
        if not verbose :
            gp_m.Params.LogToConsole = 0
        gp_x = gp_m.addMVar((self.x_len,), lb=_x, ub=x_, name="x")
        gp_u = gp_m.addMVar((self.u_len,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u")
        pred_constr = gml.add_predictor_constr(gp_m, self.nn.seq, gp_x, gp_u)
        objvars = gp_u - self._C@gp_x

        if verbose:
            print('C: ', self.C)
            print('pre  _d: ', self._d)
            print('pre  d_: ', self.d_)

        for i, obj_i in enumerate(objvars) :
            gp_m.setObjective(obj_i, GRB.MINIMIZE)
            gp_m.optimize()
            self._d[i] = gp_m.objVal
            gp_m.setObjective(obj_i, GRB.MAXIMIZE)
            gp_m.optimize()
            self.d_[i] = gp_m.objVal

        if verbose:
            print('post _d: ', self._d)
            print('post d_: ', self.d_)

        self.d = get_iarray(self._d, self.d_)
    
    def __str__(self) -> str:
        return f'{str(self.nn)}, {self.mode} mode'
