from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from matplotlib import pyplot as plt
from invariance import System, LinearControl, ContinuousTimeSpec, ControlledSystem, NoDisturbance
from invariance.utils import gen_ics, numpy_to_file
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import numpy as np
import interval
from control import lqr
import sympy as sp
from mpc import MPCController

NUM_TRAJS = 80000
FILENAME = 'mpc_10_025'
PROCESSES = 10
PLOT_DATA = False

FILEPATH = 'data/' + FILENAME + datetime.now().strftime('_%Y%m%d-%H%M%S') + '.npy'

# A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
# B = np.array([[0,0],[1,0],[0,0],[0,1]])
# Q = np.array([ [5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5] ])
# R = np.array([ [1.25,0], [0,1.25] ])
A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
B = np.array([[0,0],[1,0],[0,0],[0,1]])
Q = np.array([ [10,0,0,0], [0,10,0,0], [0,0,10,0], [0,0,0,10] ])
R = np.array([ [0.25,0], [0,0.25] ])


K, S, E = lqr(A, B, Q, R)
K = -K

print('K: ', K)
L, U = np.linalg.eig(A + B@K)
print('L: ', L)
print('U: ', U)

RANGES = [
    ([-10,10],),
    ([-10,10],),
    ([-10,10],),
    ([-10,10],),
]

px, py, vx, vy = (x_vars := sp.symbols('px vx py vy'))
ux, uy = (u_vars := sp.symbols('ux uy'))
wx, wy = (w_vars := sp.symbols('wx wy'))
f_eqn = A@x_vars + B@u_vars
t_spec = ContinuousTimeSpec(0.125, 0.125)
sys = System(x_vars, u_vars, w_vars, f_eqn, t_spec)
# control = LinearControl(K)
control = MPCController(Q, R)
clsys = ControlledSystem(sys, control, dist=NoDisturbance(2))
t_span = [0, 7]
tu = t_spec.tu(*t_span)
lenuu = t_spec.lenuu(*t_span)

NUM_POINTS = NUM_TRAJS * lenuu
print("Writing to " + FILEPATH)
print("TRAJECTORIES: %d" % NUM_TRAJS)
print("CONTROL LEN : %d" % lenuu)
print("TOTAL POINTS: %d" % NUM_POINTS)


tt = t_spec.tt(*t_span)

X0 = gen_ics(RANGES, NUM_TRAJS)

X = np.ones((NUM_POINTS, 4))
U = np.ones((NUM_POINTS, 2))

# traj = clsys.compute_trajectory(*t_span, X0[0,:])
# print(traj)
# input()

def task(x) :
    try :
        traj, uu = clsys.compute_trajectory_uu (*t_span, x)
        return traj(tu).reshape(lenuu-1, -1), uu
    except Exception as e:
        print('failed... trying with new IC')
        print(e)
        return task(gen_ics(RANGES, 1)[0,:])

pool = Pool(processes=PROCESSES)

if PLOT_DATA :
    fig, axs = plt.subplots(4,4,dpi=100,figsize=[10,10])
    fig.set_tight_layout(True)
    axs = axs.reshape(-1)
    axsi = 0

for i, result in enumerate(tqdm(pool.imap_unordered(task, X0), total=NUM_TRAJS, smoothing=0)) :
    # tt, xx, uu = result
    # traj = result
    xx, uu = result
    # print(xx.shape)
    # print(uu.shape)
    # print(X.shape)

    X[i*lenuu:(i+1)*lenuu-1,:] = xx
    U[i*lenuu:(i+1)*lenuu-1,:] = uu
    if PLOT_DATA :
        axsi = (axsi + 1) % len(axs); ax = axs[axsi]; ax.clear()
        # ax = axs[0]
        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((-4,-4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((4,-4),3/1.25,lw=0,fc='salmon'))
        # points = np.array([xx[:,0],xx[:,1]]).T.reshape(-1,1,2)
        # segs = np.concatenate([points[:-1],points[1:]],axis=1)
        # lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
        # lc.set_array(tt)
        # ax.add_collection(lc)
        # ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
        # ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

        # cmap = sns.cubehelix_palette(rot=-0.4, as_cmap=True)
        points = ax.scatter(xx[:,0], xx[:,1], c=tu, cmap=plt.get_cmap('cividis'), s=1)
        ax.set_xlim([-10,10]); ax.set_ylim([-10,10])
        # ax.set_xlim([-15,15]); ax.set_ylim([-15,15])
        # fig.colorbar(points, ax=ax)
        # ax.set_title("y vs x, color t")
        plt.ion(); plt.show(); plt.pause(0.0001)
        pass

numpy_to_file(X, U, FILEPATH)
