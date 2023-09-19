import time
import numpy as np
from numpy.typing import ArrayLike
import interval
from interval import get_lu
from numpy import diag_indices_from, clip, inf, empty
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from tqdm import tqdm
import shapely.geometry as sg
import shapely.ops as so
import sympy as sp

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

def run_times (N, func, *args, **kwargs) :
    times = np.empty(N)
    disable_bar = kwargs.pop('rt_disable_bar',False)
    for n in tqdm(range(N), disable=disable_bar) :
        before = time.time()
        ret = func(*args, **kwargs)
        after = time.time()
        times[n] = (after - before)
    return ret, times

def uniform_disjoint (set, N) :
    probs = [s[1] - s[0] for s in set]; probs = probs / np.sum(probs)
    return np.array([np.random.choice([
        np.random.uniform(s[0], s[1]) for s in set
    ], p=probs) for _ in range(N)])

def gen_ics(RANGES, N) :
    X = np.empty((N, len(RANGES)))
    for i, range in enumerate(RANGES) :
        X[:,i] = uniform_disjoint(range, N)
    return X

def gen_ics_pert (x0, pert, N) :
    X = np.empty((N, len(x0)))
    for i in range(len(x0)) :
        # X[:,i] = uniform_disjoint(range, N)
        X[:,i] = np.random.uniform(x0[i]-pert[i], x0[i]+pert[i], N)
    return X

def gen_ics_iarray (x0, N) :
    X = np.empty((N, len(x0)))
    for i in range(len(x0)) :
        # X[:,i] = uniform_disjoint(range, N)
        X[:,i] = np.random.uniform(x0[i].l, x0[i].u, N)
    return X

def file_to_numpy (filenames) :
    with open('data/' + filenames[0] + '.npy', 'rb') as f :
        nploaded = np.load(f)
        X = nploaded['X']
        U = nploaded['U']
    for FILE in filenames[1:] :
        with open('data/' + FILE + '.npy', 'rb') as f :
            nploaded = np.load(f)
            X = np.append(X, nploaded['X'], axis=0)
            U = np.append(U, nploaded['U'], axis=0)
    return X,U

def numpy_to_file (X, U, filename) :
    with open(filename, 'wb') as f :
        np.savez(f, X=X, U=U)

def plot_Y_X (fig, ax, tt, XX, YY, xlim=[-15,15], ylim=[-15,15], lw=2, show_colorbar=False, show_obs=True) :
    if show_obs :
        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon'))
        ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='salmon'))
    # ax.add_patch(Rectangle((-1,-1),2,2,lw=0,fc='darkgreen'))
    # ax.add_patch(Circle((0,0),1,lw=0,fc='g'))
    # ax.add_patch(Ellipse((-3,0),2*1.8,2*2.8,lw=0,fc='r'))
    # ax.add_patch(Rectangle((4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    # ax.add_patch(Rectangle((-4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    points = np.array([XX,YY]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, lw=lw, cmap=plt.get_cmap('cividis'),zorder=0)
    # lc = LineCollection(segs, lw=lw)
    lc.set_array(tt)
    ax.add_collection(lc)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')
    if show_colorbar :
        cb = fig.colorbar(lc, ax=ax, location='right', aspect=40, fraction=0.025, pad=0)
        cb.set_ticks([0,0.25,0.5,0.75,1,1.25])
        cb.set_label('t', rotation='horizontal')
        # cb.set_ticks(list(cb.get_ticks()) + [tt[-1]])
    ax.set_title("y vs x, color t")

def plot_XY_t (ax, tt, XX, YY, dXX=None, dYY=None, dXXh=None, dYYh=None) :
    ax.plot(tt, XX, label="X", color='C0')
    ax.plot(tt, YY, label="Y", color='C1')
    if dXX is not None and dYY is not None and dXXh is not None and dYYh is not None :
        ax.plot(tt, dXX,  color='C0', label='dx', linewidth=0.5)
        ax.plot(tt, dXXh, color='C0', label='dxh',linewidth=0.5)
        ax.fill_between(tt, dXX, dXXh, color='C0', alpha=1)
        ax.plot(tt, dYY,  color='C1', label='dy', linewidth=0.5)
        ax.plot(tt, dYYh, color='C1', label='dyh',linewidth=0.5)
        ax.fill_between(tt, dYY, dYYh, color='C1', alpha=1)
    ax.legend()
    ax.set_title("x,y vs t")
def plot_PV_t (ax, tt, PP, VV) :
    ax.plot(tt, PP, label="psi", color='C2')
    ax.plot(tt, VV, label="v", color='C3')
    ax.legend()
    ax.set_title("psi,v vs t")
def plot_XYPV_t (ax, tt, SS) :
    XX, YY, PP, VV = SS
    ax.plot(tt, XX, label="X")#, color='C0')
    ax.plot(tt, YY, label="Y")#, color='C1')
    ax2 = ax.twinx()
    ax2.plot(tt, PP, label="psi", color='C2')
    ax2.plot(tt, VV, label="v", color='C3')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)
    ax.set_xlabel('t',labelpad=0)
    ax.set_ylabel('x,y',labelpad=0)
    ax2.set_ylabel('psi,v',labelpad=0)
    ax.set_title("state vs t")
    # ax.set_xlim([1,1.5])
    # ax.set_ylim([-15,15])
    # ax.set_ylim([6,10])
def plot_u_t (ax, tt, UU_acc, UU_ang) :
    ax.plot(tt, UU_acc, label='u_acc', color='C4')
    ax2 = ax.twinx()
    ax2.plot(tt, UU_ang, label='u_ang', color='C5')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)
    ax.set_ylabel('u_acc',labelpad=0)
    ax2.set_ylabel('u_ang',labelpad=0)
    ax.set_title('u_acc,u_ang vs t')
def plot_solution(fig, axs, tt, SS, UU) :
    XX, YY, PP, VV = SS
    UU_acc, UU_ang = UU
    # plot_XY_t(axs[0,0],tt,XX,YY)
    # plot_PV_t(axs[1,0],tt,PP,VV)
    plot_XYPV_t(axs[0,0],tt,SS)
    plot_Y_X (fig,axs[0,1],tt,XX,YY)
    plot_u_t (axs[1,1],tt,UU_acc,UU_ang)

def d_metzler (A, separate=True)  :
    diag = diag_indices_from(A)
    Am = clip(A, 0, inf); Am[diag] = A[diag]
    An = A - Am
    if separate :
        return Am, An
    else :
        n = A.shape[0]
        ret = empty((2*n,2*n))
        ret[:n,:n] = Am; ret[n:,n:] = Am
        ret[:n,n:] = An; ret[n:,:n] = An
        return ret

def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret

sg_box = lambda x, xi=0, yi=1 : sg.box(x[xi].l,x[yi].l,x[xi].u,x[yi].u)
sg_boxes = lambda xx, xi=0, yi=1 : [sg_box(x, xi, yi) for x in xx]

def draw_sg_union (ax, boxes, **kwargs) :
    shape = so.unary_union(boxes)
    xs, ys = shape.exterior.xy
    kwargs.setdefault('ec', 'tab:blue')
    kwargs.setdefault('fc', 'none')
    kwargs.setdefault('lw', 2)
    kwargs.setdefault('alpha', 1)
    ax.fill(xs, ys, **kwargs)

draw_iarray = lambda ax, x, xi=0, yi=1, **kwargs : draw_sg_union(ax, [sg_box(x, xi, yi)], **kwargs)
draw_iarrays = lambda ax, xx, xi=0, yi=1, **kwargs: draw_sg_union(ax, sg_boxes(xx, xi, yi), **kwargs)


def draw_iarray_3d (ax, x, xi=0, yi=1, zi=2, color='tab:blue') :
    Xl, Yl, Zl, Xu, Yu, Zu = \
        x[xi].l, x[yi].l, x[zi].l,\
        x[xi].u, x[yi].u, x[zi].u
    faces = [ \
        np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yu,Zl],[Xl,Yu,Zl],[Xl,Yl,Zl]]), \
        np.array([[Xl,Yl,Zu],[Xu,Yl,Zu],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yl,Zu]]), \
        np.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yl,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        np.array([[Xl,Yu,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yu,Zl]]), \
        np.array([[Xl,Yl,Zl],[Xl,Yu,Zl],[Xl,Yu,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        np.array([[Xu,Yl,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xu,Yl,Zu],[Xu,Yl,Zl]]) ]
    for face in faces :
        ax.plot3D(face[:,0], face[:,1], face[:,2], color='tab:blue', alpha=0.75, lw=0.75)

def draw_iarrays_3d (ax, xx, xi=0, yi=1, zi=2, color='tab:blue') :
    for x in xx :
        draw_iarray_3d(ax, x, xi, yi, zi, color)

# def draw_iarray (ax, x, xi=0, yi=1, color='tab:blue') :
#     draw_sg_union(sg_box(x, xi, yi))

def plot_iarray_t (ax, tt, x, **kwargs) :
    xl, xu = get_lu(x)
    alpha = kwargs.pop('alpha', 0.25)
    label = kwargs.pop('label', None)
    ax.fill_between(tt, xl, xu, alpha=alpha, label=label, **kwargs)
    ax.plot(tt, xl, **kwargs)
    ax.plot(tt, xu, **kwargs)


def set_columns_from_corner (corner, _A, A_) :
    # Set i-th column of A based on corner
    _Jx = np.empty_like(_A); J_x = np.empty_like(A_)
    for i in range(len(corner)) :
        if corner[i] == 0 :
            _Jx[:,i] = _A[:,i] # Use _A when cornered on lb
            J_x[:,i] = A_[:,i] # Use A_ when cornered on lb
        else :
            _Jx[:,i] = A_[:,i] # Use A_ when cornered on ub
            J_x[:,i] = _A[:,i] # Use _A when cornered on ub
    return _Jx, J_x

def jordan_canonical (A:ArrayLike, verbose:bool=False) :
    """Compute the Jordan Canonical Form of A (real Jordan Blocks)

    Args:
        A (ArrayLike): Matrix to compute the Jordan Canonical Form for
        verbose (bool, optional): Verbose printing. Defaults to False.
    """

    # A = P J P^{-1}
    # P, J = sp.Matrix(A).jordan_form()
    J,P = np.linalg.eig(A)
    J = np.diag(J)
    P = P
    P = np.array(P).astype(np.complex64)
    J = np.array(J).astype(np.complex64)
    if verbose :
        print(f'P (complex): {P}')
        print(f'J (complex): {J}')
    
    # Convert complex Jordan blocks to real 2x2 blocks
    Preal = np.real(P)
    Pimag = np.imag(P)
    Jreal = np.real(J)
    Jimag = np.imag(J)
    skip = False
    zeros = np.zeros_like(Preal[0,:])
    Poutr = np.empty_like(Preal)
    Pouti = np.empty_like(Pimag)
    Joutr = np.empty_like(Jreal)
    Jouti = np.empty_like(Jimag)
    for i in range(len(Jimag)) :
        if not skip :
            if np.iscomplex(J[i,i]) :
                print('is complex')
                Poutr[i,:] = -Preal[i,:]
                Pouti[i,:] = zeros
                Poutr[i+1,:] = zeros
                Pouti[i+1,:] = Pimag[i+1,:]
                Jouti[i+1,i] = -Jimag[i,i]
                Jouti[i,i+1] = Jimag[i,i]
                Jouti[i,i] = 0
                Jouti[i+1,i+1] = 0
                skip = True
        else :
            skip = False
    
    P = Poutr + Pouti
    J = Joutr + Jouti
    
    if verbose :
        print(f'P (real): {P}')
        print(f'J (real): {J}')

    return P, J

def all_corners (n) :
    return [tuple(reversed([(i//(2**j)) % 2 for j in range(n)])) for i in range(2**n)]

def get_corners (y) :
    y = np.asarray(y).astype(np.interval)
    corners = all_corners(len(y))
    return np.array([np.array([(y[i].l if corner[i] == 0 else y[i].u) for i in range(len(y))]) for corner in corners])
