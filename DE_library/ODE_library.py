#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:06:10 2019

@author: adamgosztolai
"""

import numpy as np

# =============================================================================
# ODE library
# =============================================================================
def fun_double_pendulum(P = {'b': 0.05, 'g': 9.81, 'l': 1.0, 'm': 1.0}):
    
    def f(t, X):
        theta1, theta2 = X
        return [theta2, -(P['b']/P['m'])*theta2 - P['g']*np.sin(theta1)]
    
    def jac(t, X):
        theta1, theta2 = X
        return [[0.0, 1.0], 
                [-P['g']*np.cos(theta1), -P['b']/P['m']]]
    
    return f, jac 


def fun_lorenz(P = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}):
    
    def f(t, X):
        x, y, z = X
        return [P['sigma']*(y - x)/P['tau'], 
                (x*(P['rho'] - z) - y)/P['tau'], 
                (x*y - P['beta']*z)/P['tau']]
    
    def jac(t, X):
        x, y, z = X
        return [[-P['sigma']/P['tau'], P['sigma']/P['tau'], 0.], 
                [(P['rho'] - z)/P['tau'], -1./P['tau'], -x/P['tau']], 
                [y/P['tau'], x/P['tau'], -P['beta']/P['tau']]]
                
    return f, jac            
    

def fun_rossler(P = {'a': 0.15, 'b': 0.2, 'c': 10.0}):
    def f(t, X):
        x, y, z = X
        return [-y - z, x + P['a']*y, P['b'] + z * (x - P['c'])]
    def jac(t, X):
        x, y, z = X
        return [[0.,      -1, -1 ],
                [1,   P['a'],  0.],
                [z,       0.,  x ]]

    return f, jac


def fun_vanderpol(P = {'mu': 1.}):
    
    def f(t, X):
        x, y = X
        return [y, (P['mu']*(1-x**2)*y - x)]
    
    def jac(t, X):
        x, y = X
        return [[0.,                1.    ], 
                [-2.*P['mu']*x*y - 1., -P['mu']*x**2]]


    return f, jac


def fun_duffing(P = {'alpha', 'beta', 'gamma', 'delta', 'omega', 'tau'}):
    
    def f(t, X):
        x, y, z = X
        return [y/P['tau'], 
                (-P['delta']*y - P['alpha']*x - P['beta']*x**3 + P['gamma']*np.cos(z))/P['tau'], 
                P['omega']]
    
    def jac(t, X):
        x, y, z = X
        return [[0., 1./P['tau'], 0.], 
                [(-P['alpha'] - 3*P['beta']*x**2)/P['tau'], -P['delta']/P['tau'], -P['gamma']*np.sin(z)/P['tau']], 
                [0., 0., 0.]] 

    return f, jac


def fun_kuramoto(P = {'k': 5, #coupling constant
                      'W': np.array([28, 19, 11, 9, 2, 4]), 
                      'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]])}
                 ):
    
    def f(t, X):     
        Xt = X[:, None]
        dX = Xt - X
#            if self.noise != None:
#                n = self.noise().astype(self.dtype)
#                phase += n
        phase = P['W'] + np.sum(P['k']*P['K']*np.sin(dX), axis=0)

        return phase

    def jac(t, X):
        Xt = X[:, None]
        dX = Xt - X
        phase = np.zeros(P['K'].shape)
        tmp = P['K']*np.cos(dX)
        tmp -= np.diag(tmp)
        phase += np.diag(np.sum(tmp, axis=0))
        phase -= tmp
        
        return phase

    return f, jac    


def fun_kuramoto_delay(P = {
                    'k': 5, #coupling constant
                    'W': np.array([28, 19, 11, 9, 2, 4]), 
                    'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]]),
                    'tau': 1}
                 ):
    
    def f(t, X):     
        Xt = X[:, None]
        dX = Xt - X
#            if self.noise != None:
#                n = self.noise().astype(self.dtype)
#                phase += n
        phase = P['W'] + np.sum(P['k']*P['K']*np.sin(dX), axis=0)

        return phase

    return f
 
    
def fun_righetti_ijspeert(P = {'a', 'alpha', 'mu', 'K', 'omega_swing', 'omega_stance'}):

    def f(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = P['omega_stance'] + (P['omega_swing'] - P['omega_stance']) / (1+np.exp(P['a']*y))            
        R = P['alpha']*(P['mu'] - x**2 - y**2)
            
        return (R*x - omega*y).tolist() + (R*y + omega*x + P['K'].dot(y)).tolist()

    def jac(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = P['omega_stance'] + (P['omega_swing'] - P['omega_stance']) / (1+np.exp(P['a']*y))            
        R = P['alpha']*(P['mu'] - x**2 - y**2)
        
        dX = np.zeros([12,12])
        dX[6:11,6:11] = P['K']
        for i in range(6):
            dX[i,i]     = - 2*P['alpha']*x[i]**2 + R[i] 
            dX[i,i+6]   = - 2*P['alpha']*x[i]*y[i] - omega[i] + (omega[i]-P['omega_stance'])**2*P['a']*np.exp(P['a']*y[i])/(P['omega_swing']-P['omega_stance'])*y[i] 
            dX[i+6,i]   = - 2*P['alpha']*x[i]*y[i] + omega[i]
            dX[i+6,i+6] = - 2*P['alpha']*y[i]*2 + R[i] - (omega[i]-P['omega_stance'])**2*P['a']*np.exp(P['a']*y[i])/(P['omega_swing']-P['omega_stance'])*x[i]
            
        return dX.tolist()

    return f, jac