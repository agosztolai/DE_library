#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:09:14 2020

@author: gosztolai
"""
from scipy.integrate import ode
import scipy.integrate
import scipy.interpolate
import sys
import numpy as np
from .ODE_library import *

# =============================================================================
# ODE solver
# =============================================================================
def simulate_ODE(whichmodel, t, x0, P=None):
    
    if P == None:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)()
    else:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)(P)
                     
    r = ode(f, jac)
#    r.set_integrator('zvode', method='bdf')
    r.set_integrator('dopri5')
    r.set_initial_value(x0, t[0])
      
    #Run ODE integrator
    x = [x0]
    xprime = [f(0.0, x0)]   
    
    for idx, _t in enumerate(t[1:]):
        r.integrate(_t)
        x.append(np.real(r.y))
        xprime.append(f(r.t, np.real(r.y)))    

    return np.array(x), np.array(xprime)

# =============================================================================
# DDE solver
# =============================================================================
class ddeVar:
    """ special function-like variables for the integration of DDEs """

    def __init__(self, g, tc=0):
        """ g(t) = expression of Y(t) for t<tc """

        self.g = g
        self.tc= tc
        # We must fill the interpolator with 2 points minimum
        self.itpr = scipy.interpolate.interp1d(
            np.array([tc-1,tc]), # X
            np.array([self.g(tc),self.g(tc)]).T, # Y
            kind='linear', bounds_error=False,
            fill_value = self.g(tc))

    def update(self, t, Y):
        """ Add one new (ti,yi) to the interpolator """

        self.itpr.x = np.hstack([self.itpr.x, [t]])
        Y2 = Y if (Y.size==1) else np.array([Y]).T
        self.itpr.y = np.hstack([self.itpr.y, Y2])
        self.itpr.fill_value = Y
        self.itpr._y = self.itpr._reshape_yi(self.itpr.y)

    def __call__(self,t=0):
        """ Y(t) will return the instance's value at time t """

        return (self.g(t) if (t<=self.tc) else self.itpr(t))


class dde(scipy.integrate.ode):
    """ Overwrites a few functions of scipy.integrate.ode"""

    def __init__(self,f,jac=None):

        def f2(t,y,args):
            return f(self.Y,t,*args)
        scipy.integrate.ode.__init__(self,f2,jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):

        scipy.integrate.ode.integrate(self,t,step,relax)
        self.Y.update(self.t,self.y)
        return self.y

    def set_initial_value(self,Y):

        self.Y = Y #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)


def ddeint(func, g, tt, fargs=None):
    """
    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by ``func`` with history function ``g``
    and potential additional arguments for the model, ``fargs``.
    Returns the values of the solution at the times given by the array ``tt``.

    Example:
    --------

    We will solve the delayed Lotka-Volterra system defined as

    For t < 0:
    x(t) = 1+t
    y(t) = 2-t

    For t > 0:
    dx/dt =  0.5* ( 1- y(t-d) )
    dy/dt = -0.5* ( 1- x(t-d) )

    Note that here the delay ``d`` is a tunable parameter of the model.

    ---

    import numpy as np

    def model(XY,t,d):
        x, y = XY(t)
        xd, yd = XY(t-d)
        return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])

    g = lambda t : np.array([1+t,2-t]) # 'history' at t<0
    tt = np.linspace(0,30,20000) # times for integration
    d = 0.5 # set parameter d
    yy = ddeint(model,g,tt,fargs=(d,)) # solve the DDE !

    """

    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g,tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    
    return np.array([g(tt[0])] + 
                    [dde_.integrate(dde_.t + dt) for dt in np.diff(tt)])


# def simulate_DDE(whichmodel, t, x0, hist, P=None):
    
#     if P == None:
#         f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)()
#     else:
#         f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)(P)
                     
#     r = ode(f, jac)
# #    r.set_integrator('zvode', method='bdf')
#     r.set_integrator('dopri5')
#     r.set_initial_value(x0, t[0])
      
#     #Run ODE integrator
#     x = [x0]
#     xprime = [f(0.0, x0)]  
     
#     theta = ddeint(self, theta0_hist, t)
    
#     for idx, _t in enumerate(t[1:]):
#         r.integrate(_t)
#         x.append(np.real(r.y))
#         xprime.append(f(r.t, np.real(r.y)))    

#     return np.array(x), np.array(xprime)


# class KuramotoModel:

#     def __call__(self, theta, t=0):
#         """Evaluate the instance of the Kuramoto Model vector field with the given attributes.

#         Parameters
#         ----------
#         theta : ndarray of shape (N)
#             The phase value of the N oscillators in the network. It must be a function if tau>0.
#         t: float
#             The time variable. This parameter allows compatibility with odeint and ddeint from scipy.integrate.
#         tau: float
#             Overall time delay among nodes. This parameter is included here to allow compatiblitly with ddeint.

#         Returns
#         -------
#         d_theta : ndarray of shape (N)
#             The evaluated value, that is, d_theta = f(theta, t)

#         """
#         if self.tau == 0:
#             theta_t, theta_tau = theta, theta
#         # else:
#         #     theta_t, theta_tau = theta(t), theta(t - self.tau)

#         for i in range(0, self.N):
#             self.d_theta[i] = self.w[i] + \
#                               (self.c / self.d[i]) * sum(np.sin(theta_tau[j] - theta_t[i]) for j in self.A[i])

#         return self.d_theta


#     # def integrate(self, theta0, tf=100, h=0.01):
#         """Numerical integation of the Kuramoto Model with odeint (tau=0) or ddeint (tau>0).

#         Parameters
#         ----------
#         theta0 : ndarray of shape (N)
#             Initial condition. If tau>0, (lambda t: theta0 - t * self.w) is used as initial condition.
#         tf : float
#             The end of the integration interval. It starts at t=0.
#         h : float
#             Numerical integration step.

#         Returns
#         -------
#         t : ndarray
#             Time discretization points, t = [0, h, ..., tf-h, tf].
#         theta : ndarray of shape (len(t), N)
#             Numerical solution for each time in t.
#         """

        # t = np.arange(0, tf + h / 2, h)
        # if self.tau == 0:
            # theta = odeint(self, theta0, t, hmax=h)
        # else:
        #     theta0_hist = lambda t: theta0 + t * self.w
        #     theta = ddeint(self, theta0_hist, t)

        # return t, theta


# x0 = np.array([0.91713783, 4.6580943 , 0.94560485, 4.54832193, 5.88608189,
#        1.14013483, 3.04921813, 0.57283337, 4.53576829, 6.15318784,
#        4.03473894])

# W = np.array([-3.141593, -2.688248, -2.622932, -2.580255, -1.721075, -0.935027,
#         2.128966,  2.444749,  2.963254,  3.042909,  3.141593])

# E = np.array([[ 0.,  2.],
#        [ 0.,  3.],
#        [ 0.,  4.],
#        [ 0.,  6.],
#        [ 1.,  2.],
#        [ 1.,  5.],
#        [ 1.,  6.],
#        [ 1.,  7.],
#        [ 2.,  1.],
#        [ 2.,  0.],
#        [ 2., 10.],
#        [ 2.,  8.],
#        [ 2.,  5.],
#        [ 2.,  4.],
#        [ 2.,  3.],
#        [ 2.,  7.],
#        [ 3.,  0.],
#        [ 3.,  2.],
#        [ 4.,  2.],
#        [ 4.,  0.],
#        [ 5.,  2.],
#        [ 5.,  1.],
#        [ 6.,  1.],
#        [ 6.,  9.],
#        [ 6.,  0.],
#        [ 7.,  2.],
#        [ 7.,  8.],
#        [ 7.,  9.],
#        [ 7.,  1.],
#        [ 8.,  7.],
#        [ 8.,  2.],
#        [ 9.,  6.],
#        [ 9., 10.],
#        [ 9.,  7.],
#        [10.,  2.],
#        [10.,  9.]])

# ind = np.array([ 0,  5,  1,  2,  3,  6,  4,  7,  9, 10,  8])

# import networkx as nx
# G = nx.Graph()
# G.add_edges_from(E)
# A = nx.adjacency_matrix(G).toarray()
# A = A[:,ind][ind]

# K = A.dot(np.diag((1.0 / np.array([4, 4, 8, 2, 2, 2, 3, 4, 2, 3, 2]))))

# par = {'oscN': 11, 'k': 7.5, 'W': W, 'K': K}
# t = np.arange(0, 100 + 0.01 / 2, 0.01)

# phi = simulate_ODE('kuramoto', t, x0=x0, P=par)

#T = 1 #period
#num_periods = 10
#max_rate_exp = 8
#max_sampl_rate = 2**max_rate_exp
#oscN = 3 # num of oscillators
#harmN = 0 #number of harmonics in coupling matrix
#k = 5 #coupling constant
#
#_x0 = np.array([1,2,3, 1, 5, 2, 3])
#_W = np.array([28,19,11,9, 2, 4])
#_K = np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
#               [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
#               [-0.5, -0.5,  0,    1,   -0.5, -0.5],
#               [-0.5, -0.5,  1,    0,   -0.5, -0.5],
#               [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
#               [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]])
#
#_K2 = np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
#                [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
#                [-0.5, -0.5,  0,    1,   -0.5, -0.5],
#                [-0.5, -0.5,  1,    0,   -0.5, -0.5],
#                [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
#                [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]])
#
#_K = np.dstack((_K, _K2)).T
#W = _W[:oscN]
#K = k*np.squeeze(_K[:harmN+1,:oscN,:oscN])
#
#par = (W, K) 
#x0 = _x0[:oscN]
#t = np.linspace(0, num_periods*T, int(num_periods*max_sampl_rate))
#
#
##phi = simulate(fun, t, x0=x0, args = par)[0]
#
#import sys
#sys.path.append('/Users/adamgosztolai/Dropbox/github/model_discovery/SINDy/utils')
#from SINDy_library import pool_data
#
#
#from scipy.special import binom
#
#    
#Xi_true = np.zeros([1 + (harmN+1)*int(oscN*binom(oscN,oscN-1)), oscN])
#Xi_true[0,:] = W
#for i in range(oscN):
#    Xi_true[1+i*oscN:1+(i+1)*oscN,i] = K[i]
#
#variables = pool_data(x0, poly_order=0, use_sine=False, harm_coupling = True, include_constant=True, varname='x')[0]
#
#np.dot(Xi_true.T, variables)
#
#jac(t,x0,args=par)

#sol = simulate('kuramoto', np.arange(0,5), x0, args=par)
