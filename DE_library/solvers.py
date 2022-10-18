#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
from scipy.integrate import odeint
from DE_library import ODE_library

def simulate_ODE(whichmodel, t, X0, par=None, **noise_pars):
    """
    Load ODE functions and run appropriate solver

    Parameters
    ----------
    whichmodel : string
        ODE system from ODE_library.py.
    t : array or list
        Time steps to evaluate system at.
    x0 : array or list
        Initial condition. Size must match the dimension of the ODE system.
    P : dict, optional
        Parameters. The default is None.

    Returns
    -------
    x : list
        Solution.
    xprime : list
        Time derivative of solution.

    """
    
    f, jac = ODE_library.load_ODE(whichmodel, par=None)
    X = solve_ODE(f, jac, t, X0)
    
    if noise_pars!={}:
        X = addnoise(X, **noise_pars)
    
    Xprime = derivative(f, t, X)
    
    return X, Xprime


def addnoise(X, **noise_pars):
    """
    Add noise to trajectories

    Parameters
    ----------
    X : np array
        Trajectories.
    **noise_pars : additional keyword argument to specify noise parameters

    Returns
    -------
    X : len(t)xlen(X0) numpy array
        Trajectory.

    """
    
    if noise_pars['noise']=='Gaussian':
        mu = noise_pars['mu']
        sigma = noise_pars['sigma']
        X += np.random.normal(mu, sigma, size = X.shape)
        
    return X


def generate_trajectories(whichmodel, n, t, X0_range, par=None, stack=True, transient=None, seed=None, **noise_pars):
    """
    Generate an ensemble of trajectories from different initial conditions, 
    chosen randomly from a box.

    Parameters
    ----------
    whichmodel : string
        ODE system from ODE_library.py.
    n : int
        Number of trajectories to generate.
    t : array or list
        Time steps to evaluate system at..
    X0_range : list[list] e.g. [[0,1],[0,1],[0,1]]
        Lower/upper limits in each dimension of the box of initial solutions.
    par : dict, optional
        Parameters. The default is None.
    stack : bool, optional
        Stack trajectories into a long array. The default is True.
    transient : float between 0 and 1
        Ignore first transient fraction of time series.
    seed : int, optional
        Seed of random initial solutions. The default is None.
    noise : bool, optional
        Add noise to trajectories. Default is False.
    **noise_pars : additional keyword argument to specify noise parameters 

    Returns
    -------
    X_ens : numpy array
        Trajectories.
    t_ens : numpy array
        Time indices

    """
    
    if seed is not None:
        np.random.seed(seed)
        
    t_ens, X_ens = [], []
    for i in range(n):
        X0 = []
        for r in X0_range:
            X0.append(np.random.uniform(low=r[0], high=r[1]))
            
        X_ens.append(simulate_ODE(whichmodel, t, X0, par=par, **noise_pars))
        t_ens.append(np.arange(len(t)))
        
        if transient is not None:
            l_tr = int(len(X_ens[-1])*transient)
            X_ens[-1] = X_ens[-1][l_tr:]
            t_ens[-1] = t_ens[-1][:-l_tr]
        
    if stack:
        X_ens = np.vstack(X_ens)
        t_ens = np.hstack(t_ens)
        
    return t_ens, X_ens


def sample_trajectories(X, n, T, t0=0.1, seed=None):
    """
    Randomly sample trajectories from the attractor.

    Parameters
    ----------
    X : numpy array
        Trajectory including transient.
    n : int
        Number of trajectories to sample.
    T : int
        Length of trajectories (timesteps).
    t0 : float
        Initial transient fraction of the time series. The default is 0.1 (10%).
    seed : int, optional
        Seed of random initial solutions. The default is None.

    Returns
    -------
    t_sample : list(list)
        Time indices in the original attrator
    X_sample : list[array]
        n sampled trajectories.

    """
    
    if seed is not None:
        np.random.seed(seed)
    
    #Discard transient
    ind = np.arange(X.shape[0])
    ind = ind[int(t0*len(ind)):len(ind)-T]
    ts = np.random.choice(ind, size=n, replace=True)
    
    X_sample = generate_flow(X, ts, T=T)
    
    t_sample = []
    for i in range(n):
        t_sample+=list(np.arange(0,T))
        
    return t_sample, X_sample


def generate_flow(X, ts, T):
    """
    Obtain trajectories of between timepoints.

    Parameters
    ----------
    X : np array
        Trajectories.
    ts : int or np array or list[int]
        Source timepoint.
    T : int or list[int]
        End of trajectory or time horizon.

    Returns
    -------
    X_sample : list[np array].
        Set of flows of length T.

    """
    
    ts = ma.array(ts, dtype=int)
    
    if isinstance(T, int):
        tt = ma.array([ts[i]+T for i in range(len(ts))])
        tt = ma.array(tt, mask=ts.mask, dtype=int)
    else:
        tt = ma.array(T)
        assert len(tt)==len(ts), 'Number of source points must equal to the \
            number of target points.'
    
    X_sample = []
    for s,t in zip(ts,tt):
        if not ma.is_masked(s) and not ma.is_masked(t):
            X_sample.append(X[s:t+1])

    return X_sample, ts[~ts.mask], tt[~tt.mask]


def solve_ODE(f, jac, t, X0):
    
    X = odeint(f, X0, t, Dfun=jac, tfirst=True)
    
#     r = ode(f, jac)
# #    r.set_integrator('zvode', method='bdf')
#     r.set_integrator('dopri5')
#     r.set_initial_value(x0, t[0])
      
#     #Run ODE integrator
#     x = [x0]
#     xprime = [f(0.0, x0)]
    
#     for idx, _t in enumerate(t[1:]):
#         r.integrate(_t)
#         x.append(np.real(r.y))
#         xprime.append(f(r.t, np.real(r.y)))    

    return X

def derivative(f, t, X):
    Xprime = []
    for i, t_ in enumerate(t):
        Xprime.append(f(t_, X[i]))
    
    return np.vstack(Xprime)