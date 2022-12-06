#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint, ode
from DE_library import ODE_library


def solve_ODE(f, jac, t, x0, solver='standard'):
        
    if solver=='standard':
        x = odeint(f, x0, t, Dfun=jac, tfirst=True)
        xprime = [f(t_,x_) for t_, x_ in zip(t,x)]
        
    elif solver=='zvode':
        r = ode(f, jac)
        r.set_integrator('zvode', method='bdf')
        # r.set_integrator('dopri5')
        r.set_initial_value(x0, t[0])
          
        #Run ODE integrator
        x = [x0]
        xprime = [f(0.0, x0)]
        
        for idx, _t in enumerate(t[1:]):
            r.integrate(_t)
            x.append(np.real(r.y))
            xprime.append(f(r.t, np.real(r.y)))
        
    return np.vstack(x), np.vstack(xprime)


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
    par : dict, optional
        Parameters. The default is None.

    Returns
    -------
    X : list
        Solution.
    Xprime : list
        Time derivative of solution.

    """
    
    f, jac = ODE_library.load_ODE(whichmodel, par=par)
    X, Xprime = solve_ODE(f, jac, t, X0)
    
    if noise_pars!={}:
        X = addnoise(X, **noise_pars)
        
    return X, Xprime


def simulate_phase_portrait(whichmodel, X0_range, n=100, par=None, **noise_pars):
    """
    Compute phase portrait over a grid

    Parameters
    ----------
    Same as in simulate_ODE(), except:
    X0_range : list(list)
        List of initial conditions.

    Returns
    -------
    X_list : list(list)
        Solution for all trajectories.
    Xprime_list : list
        Time derivative of solution for all trajectories.

    """
    
    assert len(X0_range)==2, 'Works only in 2D.'
    
    X_list, Xprime_list = [], []
    f, jac = ODE_library.load_ODE(whichmodel, par=par)
    xs_grid = np.linspace(X0_range[0][0], X0_range[1][0], n + 1)
    ys_grid = np.linspace(X0_range[0][1], X0_range[1][1], n + 1)
    xs = (xs_grid[1:] + xs_grid[:-1]) / 2
    ys = (ys_grid[1:] + ys_grid[:-1]) / 2
    X, Y = np.meshgrid(xs, ys)
        
    for x, y in zip(X.flatten(), Y.flatten()):
        X_list.append(np.hstack([x, y]))
        Xprime_list.append(f(0, np.hstack([x, y])))
        
    X_list = np.array(X_list).reshape(n,n,2)
    X_list = [X_list[...,0], X_list[...,1]]
    Xprime_list = np.array(Xprime_list).reshape(n,n,2)
    Xprime_list = [Xprime_list[...,0], Xprime_list[...,1]]
    
    return np.array(X_list), np.array(Xprime_list)


def simulate_trajectories(whichmodel, X0_range, t=1, par=None, **noise_pars):
    """
    Compute a number of trajectories from the given initial conditions.

    Parameters
    ----------
    Same as in simulate_ODE(), except:
    X0_range : list(list)
        List of initial conditions.

    Returns
    -------
    X_list : list(list)
        Solution for all trajectories.
    Xprime_list : list
        Time derivative of solution for all trajectories.

    """
    
    X_list, Xprime_list = [], []    
    for X0 in X0_range:
        X, Xprime = simulate_ODE(whichmodel, t, X0, par=par, **noise_pars)
        X_list.append(X)
        Xprime_list.append(Xprime)
    
    return X_list, Xprime_list
    


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