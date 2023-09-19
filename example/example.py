import warnings
warnings.filterwarnings("ignore")
import numpy as np
import interval
from interval import from_cent_pert, get_lu, get_cent_pert
import torch
import torch.nn as nn
import sympy as sp
from invariance.inclusion import standard_ordering, two_orderings
from invariance.time import *
from invariance.system import *
from invariance.reach import UniformPartitioner, CGPartitioner
from invariance.control import ConstantDisturbance, UniformDisturbance, LinearControl
from invariance.utils import run_times, draw_iarray, plot_iarray_t
from invariance import InvariantSetLocator, Paralleletope, EulerSetting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from inspect import getsource
import os


def platoon (N, mode, plotting, notex, rset=0, kp=6, kv=7, u_lim=20.0) :
    os.system(r"mkdir -p figures/pngs")
    os.system(r"rm figures/pngs/*")
    x_vars = []
    u_vars = []
    w_vars = []
    f_eqn = []
    x = []

    u_softmax = lambda x : u_lim*sp.tanh(x/u_lim)
    # u_softmax = lambda x : x

    for i in range (N) :
        px_i, vx_i, py_i, vy_i = (x_i := sp.symbols(f'px_{i} vx_{i} py_{i} vy_{i}'))
        x.append(x_i)
        x_vars.extend(list(x_i))
    
    for i in range (N) :
        px_i, vx_i, py_i, vy_i = x[i]
        wx_i, wy_i = (w_i := sp.symbols(f'wx_{i} wy_{i}'))

        ux_i = 0
        uy_i = 0

        if i == 0 :
            # Only first unit is controlled by NN
            u1, u2 = (u := sp.symbols('u1 u2'))
            u_vars.extend(list(u))
            ux_i += u1
            uy_i += u2
            neighbors = []
            # neighbors = [1,N-1]
        elif i < N-1:
            # Middle units communicate with nearest two
            neighbors = [i-1, i+1]
            # neighbors = [i-1]
        elif i == N-1 :
            # Final unit communicates with nearest one
            neighbors = [i-1]

        for j in neighbors :
            print(i,j)
            vmag_j = sp.sqrt(x[j][1]**2 + x[j][3]**2)
            px_des = x[j][0] - rset*(x[j][1] / vmag_j)
            vx_des = x[j][1]
            py_des = x[j][2] - rset*(x[j][3] / vmag_j)
            vy_des = x[j][3]
            ux_i += kp*(px_des - x[i][0]) + kv*(vx_des - x[i][1])
            uy_i += kp*(py_des - x[i][2]) + kv*(vy_des - x[i][3])
        
        f_i = [
            vx_i,
            u_softmax(ux_i) + wx_i,
            vy_i,
            u_softmax(uy_i) + wy_i
        ]
        
        w_vars.extend(list(w_i))
        f_eqn.extend(f_i)
    
    print("Number of states: ", len(x_vars))

    t_spec = ContinuousTimeSpec(0.1,0.1)
    sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)

    net = NeuralNetwork('models/100r100r2_MPC')
    # net = NeuralNetwork('models/linear', False)
    # K = torch.zeros((2,4))
    # K[0,0] = -kp; K[0,1] = -kv; K[1,2] = -kp; K[1,3] = -kv
    # net.seq[0].weight = nn.Parameter(K)
    # net.seq[0].bias = nn.Parameter(torch.zeros(2))

    net.seq.insert(0,nn.Linear(len(x_vars), 4))
    W = torch.zeros((4,len(x_vars)))
    b = torch.zeros(4)
    W[0,0] = 1; W[1,1] = 1; W[2,2] = 1; W[3,3] = 1
    net.seq[0].weight = nn.Parameter(W)
    net.seq[0].bias = nn.Parameter(b)

    # dist = UniformDisturbance([np.interval(0) for w in w_vars])
    dist = UniformDisturbance([np.interval(-0.01,0.01) for w in w_vars])
    clsys = NNCSystem(sys, net, NNCSystem.InclOpts(mode), dist=dist)
    clsys.set_standard_ordering()
    clsys.set_four_corners()

    inv = InvariantSetLocator(clsys)
    init_opts = InvariantSetLocator.Opts(
        initial_pert=np.hstack((
            np.interval(-0.06,0.06),
            np.interval(-0.06,0.06),
            np.interval(-0.06,0.06),
            np.interval(-0.06,0.06),
            np.interval(-0.25,0.25),
            np.interval(-0.25,0.25),
            np.interval(-0.325,0.325),
            np.interval(-0.325,0.325),
        )),
        T_pert=np.array([
            np.interval(-0.05,0.05),
            np.interval(-0.05,0.05),
            np.interval(-0.05,0.05),
            np.interval(-0.05,0.05),
            np.interval(-0.08,0.08),
            np.interval(-0.08,0.08),
            np.interval(-0.11,0.11),
            np.interval(-0.11,0.11),
        ]),
        verbose=plotting,
        first_order_method='CROWN',
        l2reg_N=10000,
        use_Arm=False,
        milp_verbose=False,
        l2reg_tau=0.01,
    )

    if plotting :
        if not notex :
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica",
                "font.size": 14
            })
        fig, axs = plt.subplots(2,2,figsize=[8,6],dpi=100)
        fig.subplots_adjust(left=0.1,bottom=0.1,right=0.975,top=0.975,hspace=0.25,wspace=0.275)
        axs[0,0].set_xlabel(r'$p^\mathrm{L}_x$',labelpad=0)
        axs[0,0].set_ylabel(r'$p^\mathrm{L}_y$',labelpad=2,rotation=0)
        axs[0,1].set_xlabel(r'$v^\mathrm{L}_x$',labelpad=0)
        axs[0,1].set_ylabel(r'$v^\mathrm{L}_y$',labelpad=2,rotation=0)
        axs[1,0].set_xlabel(r'$p^\mathrm{F}_x$',labelpad=0)
        axs[1,0].set_ylabel(r'$p^\mathrm{F}_y$',labelpad=2,rotation=0)
        axs[1,1].set_xlabel(r'$v^\mathrm{F}_x$',labelpad=0)
        axs[1,1].set_ylabel(r'$v^\mathrm{F}_y$',labelpad=2,rotation=0)
        plt.ion(); plt.pause(1); plt.show()
        # plt.savefig('figures/monotone.pdf')

        def plot_paralleletope (paralleletope:Paralleletope, resize=False, i=None, **kwargs) :
            paralleletope.plot_2d_projection(axs[0,0], 0, 2, resize=resize, **kwargs)
            paralleletope.plot_2d_projection(axs[0,1], 1, 3, resize=resize, **kwargs)
            paralleletope.plot_2d_projection(axs[1,0], 4, 6, resize=resize, **kwargs)
            paralleletope.plot_2d_projection(axs[1,1], 5, 7, resize=resize, **kwargs)
            i_str = f"_{i}" if i is not None else ""
            plt.draw(); plt.pause(0.25); plt.savefig(f'figures/pngs/monotone{i_str}.png',dpi=400)
            # plt.show(block=False)

        plt.savefig('figures/monotone.pdf')

        t_start = time.time()
        result = inv.compute_invariant_paralleletope_family (init_opts, EulerSetting(0.1, 90), 
                                                            EulerSetting(0.05, 10), plot_paralleletope,
                                                            True)
        t_end = time.time()

        os.system(r"rm figures/monotone.mp4 | ffmpeg -framerate 25 -i figures/pngs/monotone_%d.png figures/monotone.mp4")
        plt.ioff()
        plt.savefig('figures/monotone.pdf')
    else :
        t_start = time.time()
        result = inv.compute_invariant_paralleletope_family (init_opts, EulerSetting(0.1, 90), 
                                                            EulerSetting(0.05, 10))
        t_end = time.time()

 
    # result = inv.compute_invariant_paralleletope_family (init_opts, 0.15, 10, plot_paralleletope)
    # result = inv.compute_invariant_paralleletope_family (init_opts, 0.15, 1, plot_paralleletope)

    # paralleletope = result.paralleletopes[-1]
    print('\n\n===== Results =====')
    _G, G_ = result.results[0].G
    print(f"Initial _G: {_G}")
    print(f"Initial G_: {G_}")
    print(f"Initial runtime: {result.results[0].runtime}")
    _G, G_ = result.results[-1].G
    print(f"Final _G: {_G}")
    print(f"Final G_: {G_}")
    print(f"Total runtime: {t_end - t_start}")
    print(f"Size of family: {len(result.paralleletopes)}")

    if plotting:
        plt.show()

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="Platooning Experiments")
    parser.add_argument('--runtime', help="Disable plotting for accurate runtime estimation", 
                        action='store_true', default=False)
    parser.add_argument('--notex', help="Disable LaTeX rendering on figures", 
                        action='store_true', default=False)
    parser.add_argument('--mode', help="Inclusion mode",
                        type=str, default='jacobian')
    args = parser.parse_args()

    from tabulate import tabulate

    table = [['$N$ (units)', 'States', 'Runtime (s)']]
    # for N in [1,4,9,20,50] :
    #     runtimes = platoon(N, False, args.runtime_N, args.mode)
    #     table.append([N, 4*N, rf'${np.mean(runtimes)} \pm {np.std(runtimes)}$'])

    # print(tabulate(table,tablefmt="latex_raw"))

    platoon(2, args.mode, not args.runtime, args.notex)
    # platoon(50, False, 1, args.mode)
