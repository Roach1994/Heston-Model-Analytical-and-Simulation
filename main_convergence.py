# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:55:22 2018

@author: 51593
"""

# -*- coding: utf-8 -*-
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ql_exact
import Euler
import KahlJackal
import Marsaglia
import Milstein
from scipy.optimize import fmin
from scipy import *
analytical=ql_exact.call_price_exact
EulerHeston=Euler.EulerHeston
MilsteinHeston=Milstein.MilsteinHeston
KJHeston=KahlJackal.KJHeston
GMHeston=Marsaglia.GMHeston
T = 10
#risk free rate
r = 0.00
#long term volatility(equiribrium level)
theta = 0.04
#Mean reversion speed of volatility
kappa = 0.5
#beta(volatility of Volatility)
beta = 1
#rho
rho = -0.9
#Initial stock price
s0 = 100.0
#Initial volatility
v0 = 0.04
#case2
#T = 5
#r = 0.05
#theta = 0.09
#kappa = 1
#beta = 1
#rho = -0.3
#s0 = 100.0
#v0 = 0.09
K = np.arange(60, 180, 40)
K=100
N=100000
dt_range=np.arange(1, 1/32, -1/32)
result=np.zeros((4,len(dt_range)))
result_std=np.zeros((4,len(dt_range)))
real=analytical(kappa, theta, beta, rho, v0, r, T, s0, K)
for i in range(0,len(dt_range)):
    dt=dt_range[i].item()  
    eul=EulerHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    mil=MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    KJ=KJHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    GM=GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    result[:,i]=[eul[0]-real,mil[0]-real,KJ[0]-real,GM[0]-real]
    result_std[:,i]=[eul[1],mil[1],KJ[1],GM[1]]
    #convert int32 to int
s = pd.Series(['Eul','Mil','KJ','GM'])
df = pd.DataFrame(result, index=s, columns=dt_range)
x=-log(dt_range)
for i in range(0,4):
    plt.plot(x,abs(result[i,:]),label=s[i])
plt.xlabel("-ln(dt)")
plt.ylabel("Error")
plt.legend()
plt.show()
fig2 = plt.figure()
for i in range(0,4):
    plt.plot(x,abs(result_std[i,:]),label=s[i])
plt.xlabel("-ln(dt)")
plt.ylabel("Std")
plt.legend()
plt.show()