#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:02:04 2021

@author: gosztola
"""
import numpy as np
from DE_library import simulate_ODE
import matplotlib.pyplot as plt

t0, t1, dt = 0, 40, 0.05
t = np.arange(t0, t1, dt)

rng = np.random.RandomState(1)
oscN = 5
x0 = rng.rand(5)#np.array([0, np.pi, 0, 1, 5, 2])
W = rng.rand(5)#np.array([28, 19, 11, 9, 2, 4])
K = rng.rand(5,5)


par = {'W': W, 'K': 1*K}

phi, d_phi = simulate_ODE('kuramoto', t, x0=x0, P = par)

plt.plot(d_phi)
plt.show()