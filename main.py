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
T = 5
r = 0.05
theta = 0.09
kappa = 1
beta = 1
rho = -0.3
s0 = 100.0
v0 = 0.09
K = np.arange(60, 180, 40)
result=np.zeros((5,3))
result_std=np.zeros((4,3))
result_time=np.zeros((4,3))
N=100000
dt=1/8
for i in range(0,3):
    real=analytical(kappa, theta, beta, rho, v0, r, T, s0, K[i].item())
    eul=EulerHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    mil=MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    KJ=KJHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    GM=GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    result[:,i]=[real,eul[0],mil[0],KJ[0],GM[0]]
    result_std[:,i]=[eul[1],mil[1],KJ[1],GM[1]]
    result_time[:,i]=[eul[2],mil[2],KJ[2],GM[2]]
    #convert int32 to int
s = pd.Series(['Real','Eul','Mil','KJ','GM'])
df = pd.DataFrame(result, index=s, columns=K)
print(df)
df2 = pd.DataFrame(result_std, index=s.iloc[1:5], columns=K)
print(df2)
df3 = pd.DataFrame(result_time, index=s.iloc[1:5], columns=K)
print(df3)