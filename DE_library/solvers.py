#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
from DE_library import ODE_library

def simulate_ODE(whichmodel, t, X0, P=None):
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
    
    f, jac = ODE_library.load_ODE(whichmodel, P=None)
    X = solve_ODE(f, jac, t, X0)
    
    Xprime = derivative(f, t, X)
    
    return X, Xprime

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