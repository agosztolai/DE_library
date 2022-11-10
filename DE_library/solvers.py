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


def simulate_phase_portrait(whichmodel, t, X0_range, par=None, **noise_pars):
    """
    Compute phase portrait by generating n trajectories of length T.

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