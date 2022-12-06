#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import numpy as np

def time_series(T, 
                X, 
                style='o', 
                node_feature=None, 
                figsize=(10,5), 
                lw=1, 
                ms=5,
                save=None):
    """
    Plot time series.

    Parameters
    ----------
    X : np array or list[np array]
        Trajectories.
    style : string
        Plotting style. The default is 'o'.
    color: bool
        Color lines. The default is True.
    lw : int
        Line width.
    ms : int
        Marker size.

    Returns
    -------
    ax : matplotlib axes object.

    """
            
    if not isinstance(X, list):
        X = [X]
            
    fig = plt.figure(figsize=figsize, constrained_layout=True)  
    grid = gridspec.GridSpec(len(X), 1, wspace=0.5, hspace=0, figure=fig)
    
    for sp, X_ in enumerate(X):
        
        if sp == 0:
            ax = plt.Subplot(fig, grid[sp])
        else:
            ax = plt.Subplot(fig, grid[sp], sharex=ax)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        if sp < len(X)-1:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('none') 
        
        colors = set_colors(node_feature)[0]
                
        for i in range(len(X_)-2):
            if X_[i] is None:
                continue
            
            c = colors[i] if len(colors)>1 and not isinstance(colors,str) else colors
                    
            ax.plot(T[i:i+2], X_[i:i+2], style, c=c, linewidth=lw, markersize=ms)
            
            fig.add_subplot(ax)
        
    if save is not None:
        savefig(fig, save)
        
    return ax


def trajectories(X,
                 V,
                 ax=None, 
                 style='o', 
                 node_feature=None, 
                 lw=1, 
                 ms=5, 
                 arrowhead=1, 
                 arrow_spacing=3,
                 axis=False, 
                 alpha=None):
    """
    Plot trajectory in phase space. If multiple trajectories
    are given, they are plotted with different colors.

    Parameters
    ----------
    X : np array
        Positions.
    V : np array
        Velocities.
    style : string
        Plotting style. The default is 'o'.
    node_feature: bool
        Color lines. The default is None.
    lw : int
        Line width.
    ms : int
        Marker size.

    Returns
    -------
    ax : matplotlib axes object.

    """
            
    dim = X.shape[1]
    assert dim==2 or dim==3, 'Dimension must be 2 or 3.'
    
    if ax is None:
        _, ax = create_axis(dim)
            
    c = set_colors(node_feature)[0]
    if alpha is not None:
        al=np.ones(len(X))*alpha
    elif len(c)>1 and not isinstance(c, str):
        al=np.abs(node_feature)/np.max(np.abs(node_feature))
    else:
        al=1
                
    if dim==2:
        if 'o' in style:
            ax.scatter(X[:,0], X[:,1], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:,0], X[:,1], c=c, linewidth=lw, markersize=ms, alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(color=c, alpha=al, lw=lw)
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            for j in range(X.shape[0]):
                ax.quiver(X[j,0], X[j,1], V[j,0]*0.1, V[j,1]*0.1,
                          **arrow_prop_dict)
    elif dim==3:
        if 'o' in style:
            ax.scatter(X[:,0], X[:,1], X[:,2], c=c, s=ms, alpha=al)
        if '-' in style:
            ax.plot(X[:,0], X[:,1], X[:,2], c=c, linewidth=lw, markersize=ms, alpha=al)
        if '>' in style:
            arrow_prop_dict = dict(mutation_scale=arrowhead, arrowstyle='-|>', color=c, alpha=al, lw=lw)
            skip = (slice(None, None, arrow_spacing), slice(None))
            X, V = X[skip], V[skip]
            for j in range(X.shape[0]):
                a = Arrow3D([X[j,0], X[j,0]+V[j,0]], 
                            [X[j,1], X[j,1]+V[j,1]], 
                            [X[j,2], X[j,2]+V[j,2]], 
                            **arrow_prop_dict)
                ax.add_artist(a)
                
    if not axis:
        ax = set_axes(ax, off=True)
        
    return ax    


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

        

# =============================================================================
# Helper functions
# =============================================================================
def create_axis(*args, fig=None):
    
    dim = args[0]
    if len(args)>1:
        args = [args[i] for i in range(1,len(args))]
    else:
        args = (1,1,1)
    
    if fig is None:
        fig = plt.figure()
        
    if dim==2:
        ax = fig.add_subplot(*args)
    elif dim==3:
        ax = fig.add_subplot(*args, projection="3d")
        
    return fig, ax


def get_limits(ax):
    lims = [ax.get_xlim(), ax.get_ylim()]
    if ax.name=="3d":
        lims.append(ax.get_zlim())
        
    return lims


def set_axes(ax, lims=None, padding=0.1, off=True):
    
    if lims is not None:
        xlim = lims[0]
        ylim = lims[1]
        pad = padding*(xlim[1] - xlim[0])
        
        ax.set_xlim([xlim[0]-pad, xlim[1]+pad])
        ax.set_ylim([ylim[0]-pad, ylim[1]+pad])
        if ax.name=="3d":
            zlim = lims[2]
            ax.set_zlim([zlim[0]-pad, zlim[1]+pad])
        
    if off:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if ax.name=="3d":
            ax.set_zticklabels([])        
    
    return ax


def set_colors(color, cmap=plt.cm.coolwarm):
    
    if color is None:
        return 'C0', None
    else:
        assert isinstance(color, (list, tuple, np.ndarray))
        
    if isinstance(color[0], (float, np.floating)):
        if (color>=0).all():
            
            norm = plt.cm.colors.Normalize(0, np.max(np.abs(color)))
        else:    
            norm = plt.cm.colors.Normalize(-np.max(np.abs(color)), np.max(np.abs(color)))
        
        colors = []
        for i, c in enumerate(color):
            colors.append(cmap(norm(np.array(c).flatten())))
   
    elif isinstance(color[0], int):
        colors = [f"C{i}" for i in np.arange(1, color.max()+1)]
        cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(1, color.max()+2), 
                                                              colors)
        
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            
    return colors, cbar


def savefig(fig, filename, folder='../results'):
    """Save figures in subfolders and with different extensions."""
    if fig is not None:
        if not Path(folder).exists():
            os.mkdir(folder)
        fig.savefig((Path(folder) / filename), bbox_inches="tight")