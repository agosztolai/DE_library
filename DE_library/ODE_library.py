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

# =============================================================================
# Codimension 1 bifurcations in 1D
# =============================================================================
def fun_saddle_node(par = {'mu': 1}):
    """parrototypical system exhibiting a saddle node bifurcation at mu=0"""

    def f(t, X):
        x, y = X
        f1 = par['mu'] - x**2
        f2 = -y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [-2*x, 0.0]
        df2 = [0.0, -1.0]

        return [df1, df2]

    return f, jac


def fun_transcritical_pitchfork(par = {'mu': 1}):
    """parrototypical system exhibiting a *transcritical* pitchfork bifurcation at mu=0"""

    def f(t, X):
        x, y = X
        f1 = par['mu']*x - x**2
        f2 = -y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [par['mu']-2*x, 0.0]
        df2 = [0.0, -1.0]

        return [df1, df2]

    return f, jac


def fun_supcritical_pitchfork(par = {'mu': 1}):
    """prototypical system exhibiting a *supercritical* pitchfork bifurcation at mu=0"""

    def f(t, X):
        x, y = X
        f1 = par['mu']*x - x**3
        f2 = -y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [par['mu']-3*x**2, 0.0]
        df2 = [0.0, -1.0]

        return [df1, df2]

    return f, jac


def fun_subcritical_pitchfork(par = {'mu': 1}):
    """prototypical system exhibiting a *subcritical* pitchfork bifurcation at mu=0"""

    def f(t, X):
        x, y = X
        f1 = par['mu']*x + x**3
        f2 = -y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [par['mu']+3*x**2, 0.0]
        df2 = [0.0, -1.0]

        return [df1, df2]

    return f, jac


# =============================================================================
# Codimension 1 bifurcations in 2D
# =============================================================================
def fun_supcritical_hopf(par = {'mu': 1., 'omega': 1., 'b': 1.}):
    """System exhibiting a *supercritical* Hopf bifurcation at mu=0, while
    keeping other parameters constant"""

    def f(t, X):
        x, y = X
        f1 = (par['mu'] - x**2 - y**2)*x - par['omega']*y
        f2 = (par['mu'] - x**2 - y**2)*y + par['omega']*x

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [par['mu'] - 3*x**2 - y**2, -2*x*y - par['omega']]
        df2 = [-2*x*y - par['omega'], par['mu'] - x**2 - 3*y**2]

        return [df1, df2]

    return f, jac


def fun_subcritical_hopf(par = {'mu': 1., 'omega': 1., 'b': 1.}):
    """prototypical system exhibiting a *subcritical* Hopf bifurcation at mu=0
    as mu increases"""

    def f(t, X):
        x, _ = X
        f1 = par['mu']*x + x**3 - x**5
        f2 = par['omega'] + par['b']*x**2

        return [f1, f2]

    def jac(t, X):
        x, _ = X
        df1 = [par['mu']+3*x**2-5*x**4, 0.0]
        df2 = [2*par['b']*x, 0.0]

        return [df1, df2]

    return f, jac


def fun_saddle_node_of_cycles(par = {'mu': -1., 'omega': 1., 'b': 1.}):
    """Saddle node bifurcation of cycles at mu=-1/4"""

    def f(t, X):
        r, _ = X
        f1 = par['mu']*r + r**3 - r**5
        f2 = par['omega'] + par['b']*r**2

        return [f1, f2]

    def jac(t, X):
        r, _ = X
        df1 = [par['mu']+3*r**2-5*r**4, 0.0]
        df2 = [2*par['b']*r, 0.0]

        return [df1, df2]

    return f, jac


def fun_infinite_period(par = {'mu'}):
    """System exhibiting an infinite-period bifurcation at mu=1"""

    def f(t, X):
        r, theta = X
        f1 = r*(1-r**2)
        f2 = par['mu'] - np.sin(theta)

        return [f1, f2]

    def jac(t, X):
        r, theta = X
        df1 = [1-r**2 - 2*r, 0.0]
        df2 = [0.0, -np.cos(theta)]

        return [df1, df2]

    return f, jac


def fun_homoclinic(par = {'mu'}):
    """System exhibiting an homoclinicbifurcation at mu=-0.8645"""

    def f(t, X):
        x, y = X
        f1 = y
        f2 = par['mu'] + x - x**2 + x*y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [0.0, 1.0]
        df2 = [1.0 - 2*x + y, x]

        return [df1, df2]

    return f, jac


def fun_vanderpol(par = {'mu': 1.}):
    """Van der parol oscillator exhibiting a degenerate Hopf bifurcation"""

    def f(t, X):
        x, y = X
        f1 = y
        f2 = par['mu']*(1-x**2)*y - x

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [0.,                     1.             ]
        df2 = [-2.*par['mu']*x*y - 1., -par['mu']*x**2]

        return [df1, df2]

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
        df1 = [0., 1./par['tau'], 0.]
        df2 = [(-par['alpha'] - 3*par['beta']*x**2)/par['tau'],
                -par['delta']/par['tau'],
                -par['gamma']*np.sin(z)/par['tau']]
        df3 = [0., 0., 0.]

        return [df1, df2, df3]

    return f, jac


def brusselator(par = {'a': 1.0, 'b': 1.0}):
    """Brusselator"""

    def f(t, X):
        x, y = X
        f1 = 1 - (par['b']+1)*x - par['a']*x**2*y
        f2 = par['b']*x + par['a']*x**2*y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [-(par['b']+1)*x - 2*par['a']*x*y, - par['a']*x**2]
        df2 = [par['b'] + 2*par['a']*x*y, par['a']*x**2]

        return [df1, df2]

    return f, jac


# =============================================================================
# Codimension 2 bifurcation in 2D
# =============================================================================
def fun_bogdanov_takens(par = {'beta1': -0.1, 'beta2': 0}):
    """Bogdanov-Takens system"""

    def f(t, X):
        x, y = X
        f1 = y
        f2 = par['beta1'] + par['beta2']*x + x**2 - x*y

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [0., 1.]
        df2 = [par['beta2'] + 2*x - y, -x]

        return [df1, df2]

    return f, jac


# =============================================================================
# Chaotic systems in 3D
# =============================================================================
def fun_double_pendulum(par = {'b': 0.05, 'g': 9.81, 'l': 1.0, 'm': 1.0}):
    """Double pendulum"""

    def f(t, X):
        x, y = X
        f1 = y
        f2 = -(par['b']/par['m'])*y - par['g']*np.sin(x)

        return [f1, f2]

    def jac(t, X):
        x, y = X
        df1 = [0.0, 1.0]
        df2 = [-par['g']*np.cos(x), -par['b']/par['m']]

        return [df1, df2]

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
        df1 = [-par['sigma']/par['tau'], par['sigma']/par['tau'], 0.]
        df2 = [(par['rho'] - z)/par['tau'], -1./par['tau'], -x/par['tau']]
        df3 = [y/par['tau'], x/par['tau'], -par['beta']/par['tau']]

        return [df1, df2, df3]

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
        df1 = [0.,      -1, -1 ]
        df2 = [1,   par['a'],  0.]
        df3 = [z,       0.,  x ]

        return [df1, df2, df3]

    return f, jac

# =============================================================================
# Higher dimensional oscillators
# =============================================================================
def fun_oscillations_on_torus(par = {'omega1','omega2','K1','K2'}):
    """Two-component Kuramoto model"""
    def f(t, X):
        t1, t2 = X
        f1 = par['omega1'] + par['K1']*np.sin(t2-t1)
        f2 = par['omega2'] + par['K2']*np.sin(t1-t2)

        return [f1, f2]

    def jac(t, X):
        t1, t2 = X
        df1 = [-par['K1']*np.cos(t2-t1)*t1, par['K1']*np.cos(t2-t1)*t2]
        df2 = [par['K2']*np.cos(t1-t2)*t1, -par['K2']*np.cos(t1-t2)*t2]

        return [df1, df2]

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

    def f(t, X):
        Xt = X[:, None]
        dX = X-Xt
        phase = par['W'].astype(float)
        phase += (par['K']*np.sin(dX)).sum(1)

        return phase

    def jac(t, X):

        Xt = X[:, None]
        dX = X-Xt

        phase = par['K']*np.cos(dX)
        phase = np.sum(phase, axis=0)

        phase = par['K']*np.cos(dX)
        phase -= np.diag(phase.sum(0))

        return phase

    return f, jac


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

# =============================================================================
# Hysteresis
# =============================================================================
def fun_cdc2_cyclin(par = {'a1':1,'a2':1,'b1':200,'b2':10,'K1':30,'K2':1,'gamma1':4,'gamma2':4,'nu':1}):
    """Cdc2-Cyclin B/Wee1 System. Exhibits hysteresis y1 steady, when varying
    \nu between 0 and 2 and back"""
    def f(t, X):
        x1, x2, y1, y2 = X
        f1 =  par['a1']*x2 - (par['b1']*x1*(par['nu']*y1)**par['gamma1'])/(par['K1']+(par['nu']*y1)**par['gamma1'])
        f2 = -par['a1']*x2 + (par['b1']*x1*(par['nu']*y1)**par['gamma1'])/(par['K1']+(par['nu']*y1)**par['gamma1'])
        f3 =  par['a2']*y2 - (par['b2']*y1*x1**par['gamma2'])/(par['K2']+x1**par['gamma2'])
        f4 = -par['a2']*y2 + (par['b2']*y1*x1**par['gamma2'])/(par['K2']+x1**par['gamma2'])

        return [f1, f2, f3, f4]

    return f, None


def fun_cdc2_cyclin_reduced(par = {'a1':1,'a2':1,'b1':200,'b2':10,'K1':30,'K2':1,'gamma1':4,'gamma2':4,'nu':1}):
    """Cdc2-Cyclin B/Wee1 System. Exhibits hysteresis y1 steady, when varying
    \nu between 0 and 2 and back"""
    def f(t, X):
        x1, y1 = X
        f1 =  par['a1']*(1-x1) - (par['b1']*x1*(par['nu']*y1)**par['gamma1'])/(par['K1']+(par['nu']*y1)**par['gamma1'])
        f2 =  par['a2']*(1-y1) - (par['b2']*y1*x1**par['gamma2'])/(par['K2']+x1**par['gamma2'])

        return [f1, f2]

    return f, None
