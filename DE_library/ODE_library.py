#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys

"""
Library of standard dynamical systems

"""

def load_ODE(whichmodel, par=None):
    """
    Load ODE system

    pararameters
    ----------
    whichmodel : sting
        ODE system from ODE_library.py.
    par : dict, optional
        pararameters. The default is None.

    Returns
    -------
    f : Callable
        ODE function.
    jac : Callable
        Jacobian.

    """

    if par == None:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)()
    else:
        f, jac = getattr(sys.modules[__name__], "fun_%s" % whichmodel)(par)
                     
    return f, jac


def fun_saddle_node(par = {'mu': 1}):
    """parrototypical system exhibiting a saddle node bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu'] - x**2
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [-2*x, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    

def fun_trans_pitch(par = {'mu': 1}):
    """parrototypical system exhibiting a *transcritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu']*x - x**2
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [par['mu']-2*x, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_sup_pitch(par = {'mu': 1}):
    """parrototypical system exhibiting a *supercritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu']*x - x**3
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [par['mu']-3*x**2, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_sub_pitch(par = {'mu': 1}):
    """parrototypical system exhibiting a *subcritical* pitchfork bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu']*x + x**3
        f2 = -y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [par['mu']+3*x**2, 0.0]
        dfdy = [0.0, -1.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_sup_hopf(par = {'mu': 1, 'omega': 1, 'b': 1}):
    """parrototypical system exhibiting a *supercritical* Hopf bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu']*x - x**3
        f2 = par['omega'] + par['b']*x**2
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [par['mu']-3*x**2, 0.0]
        dfdy = [2*par['b']*x, 0.0]
        
        return [dfdx, dfdy]
    
    return f, jac
    
    
def fun_sub_hopf(par = {'mu': 1, 'omega': 1, 'b': 1}):
    """parrototypical system exhibiting a *subcritical* Hopf bifurcation at mu=0"""
    
    def f(t, X):
        x, y = X
        f1 = par['mu']*x + x**3 - x**5
        f2 = par['omega'] + par['b']*x**2
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [par['mu']+3*x**2-5*x**4, 0.0]
        dfdy = [2*par['b']*x, 0.0]
        
        return [dfdx, dfdy]
    
    return f, jac
        

def fun_double_pendulum(par = {'b': 0.05, 'g': 9.81, 'l': 1.0, 'm': 1.0}):
    """Double pendulum"""
    
    def f(t, X):
        x, y = X
        f1 = y
        f2 = -(par['b']/par['m'])*y - par['g']*np.sin(x)
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [0.0, 1.0]
        dfdy = [-par['g']*np.cos(x), -par['b']/par['m']]
              
        return [dfdx, dfdy]
    
    return f, jac


def fun_lorenz(par = {'sigma': 10.0, 'beta': 8/3.0, 'rho': 28.0, 'tau': 1.0}):
    """Lorenz system"""
    
    def f(t, X):
        x, y, z = X
        f1 = par['sigma']*(y - x)/par['tau']
        f2 = (x*(par['rho'] - z) - y)/par['tau']
        f3 = (x*y - par['beta']*z)/par['tau']
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [-par['sigma']/par['tau'], par['sigma']/par['tau'], 0.]
        dfdy = [(par['rho'] - z)/par['tau'], -1./par['tau'], -x/par['tau']]
        dfdz = [y/par['tau'], x/par['tau'], -par['beta']/par['tau']]
        
        return [dfdx, dfdy, dfdz]
                
    return f, jac            
    

def fun_rossler(par = {'a': 0.15, 'b': 0.2, 'c': 10.0}):
    """Rossler system"""
    
    def f(t, X):
        x, y, z = X
        f1 = -y - z
        f2 = x + par['a']*y
        f3 = par['b'] + z * (x - par['c'])
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [0.,      -1, -1 ]
        dfdy = [1,   par['a'],  0.]
        dfdz = [z,       0.,  x ]
        
        return [dfdx, dfdy, dfdz]

    return f, jac


def fun_vanderpol(par = {'mu': 1.}):
    """Van der parol oscillator"""
    
    def f(t, X):
        x, y = X
        f1 = y
        f2 = par['mu']*(1-x**2)*y - x
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [0.,                     1.             ]
        dfdy = [-2.*par['mu']*x*y - 1., -par['mu']*x**2]
        
        return [dfdx, dfdy]

    return f, jac


def fun_duffing(par = {'alpha': 1., 'beta': 1., 'gamma': .1, 'delta': 2., 'omega': 1., 'tau': 1.}):
    """Duffing oscillator"""
    
    def f(t, X):
        x, y, z = X
        f1 = y/par['tau']
        f2 = (-par['delta']*y - par['alpha']*x - par['beta']*x**3 + par['gamma']*np.cos(z))/par['tau']
        f3 = par['omega']
        
        return [f1, f2, f3]
    
    def jac(t, X):
        x, y, z = X
        dfdx = [0., 1./par['tau'], 0.]
        dfdy = [(-par['alpha'] - 3*par['beta']*x**2)/par['tau'], 
                -par['delta']/par['tau'], 
                -par['gamma']*np.sin(z)/par['tau']]
        dfdz = [0., 0., 0.]

        return [dfdx, dfdy, dfdz]

    return f, jac


def fun_bogdanov_takens(par = {'beta1':-0.1, 'beta2':0}):
    """Bogdanov-Takens oscillator"""
    
    def f(t, X):
        x, y = X
        f1 = y
        f2 = par['beta1'] + par['beta2']*x + x**2 - x*y
        
        return [f1, f2]
    
    def jac(t, X):
        x, y = X
        dfdx = [1., 0.]
        dfdy = [par['beta2'] + 2*x - y, 
                -x]

        return [dfdx, dfdy]

    return f, jac


def fun_kuramoto(par = {'W': np.array([28, 19, 11, 9, 2, 4]), 
                      'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]])}    
                 ):
    """Kuramoto oscillator"""
    
#     def f(t, X):     
#         Xt = X[:, None]
#         dX = Xt - X
# #            if self.noise != None:
# #                n = self.noise().astype(self.dtype)
# #                phase += n
#         phase = par['W'] + np.sum(par['K']*np.sin(dX), axis=0)

#         return phase
    
    def f(t, X):
        Xt = X[:, None]
        dX = X-Xt
        phase = par['W'].astype(float)
        # if self.noise != None:
        #     n = self.noise().astype(self.dtype)
        #     phase += n
        phase += np.sum(par['K']*np.sin(dX),axis=1)

        return phase

    # def jac(t, X):
    #     Xt = X[:, None]
    #     dX = X - Xt
    #     phase = np.zeros(par['K'].shape)
    #     tmp = par['K']*np.cos(dX)
    #     tmp -= np.diag(tmp)
    #     phase += np.diag(np.sum(tmp, axis=0))
    #     phase -= tmp
        
    #     return phase
    
    def jac(t, X):

        Xt = X[:,None]
        dX = X-Xt
        
        # m_order = par['K'].shape[0]

        # phase = [m*par['K'][m-1]*np.cos(m*dX) for m in range(1,1+m_order)]
        phase = par['K']*np.cos(dX)
        phase = np.sum(phase, axis=0)

        for i in range(par['K'].shape[0]):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

    return f, jac    


def fun_kuramoto_delay(par = {
                    'W': np.array([28, 19, 11, 9, 2, 4]), 
                    'K': np.array([[ 0,   -0.5, -0.5, -0.5,  1,   -0.5],
                                     [-0.5,  0,   -0.5, -0.5, -0.5,  1  ],
                                     [-0.5, -0.5,  0,    1,   -0.5, -0.5],
                                     [-0.5, -0.5,  1,    0,   -0.5, -0.5],
                                     [ 1,   -0.5, -0.5, -0.5,  0,   -0.5],
                                     [-0.5,  1,   -0.5, -0.5, -0.5,  0  ]]),
                    'tau': 1}
                 ):
    """Kuramoto oscillator with delay"""
    
    def f(t, X):     
        Xt = X[:, None]
        dX = Xt - X
#            if self.noise != None:
#                n = self.noise().astype(self.dtype)
#                phase += n
        phase = par['W'] + np.sum(par['K']*np.sin(dX), axis=0)

        return phase

    return f
 
    
def fun_righetti_ijspeert(par = {'a', 'alpha', 'mu', 'K', 'omega_swing', 'omega_stance'}):
    """Righetti-Ijspeert phase oscillator"""

    def f(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = par['omega_stance'] + (par['omega_swing'] - par['omega_stance']) / (1+np.exp(par['a']*y))            
        R = par['alpha']*(par['mu'] - x**2 - y**2)
            
        return (R*x - omega*y).tolist() + (R*y + omega*x + par['K'].dot(y)).tolist()

    def jac(t, X):
        x = np.array(X[:6])
        y = np.array(X[6:])
            
        omega = par['omega_stance'] + (par['omega_swing'] - par['omega_stance']) / (1+np.exp(par['a']*y))            
        R = par['alpha']*(par['mu'] - x**2 - y**2)
        
        dX = np.zeros([12,12])
        dX[6:11,6:11] = par['K']
        for i in range(6):
            dX[i,i]     = - 2*par['alpha']*x[i]**2 + R[i] 
            dX[i,i+6]   = - 2*par['alpha']*x[i]*y[i] - omega[i] + (omega[i]-par['omega_stance'])**2*par['a']*np.exp(par['a']*y[i])/(par['omega_swing']-par['omega_stance'])*y[i] 
            dX[i+6,i]   = - 2*par['alpha']*x[i]*y[i] + omega[i]
            dX[i+6,i+6] = - 2*par['alpha']*y[i]*2 + R[i] - (omega[i]-par['omega_stance'])**2*par['a']*np.exp(par['a']*y[i])/(par['omega_swing']-par['omega_stance'])*x[i]
            
        return dX.tolist()

    return f, jac