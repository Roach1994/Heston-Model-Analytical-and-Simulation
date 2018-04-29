# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:27:22 2018

@author: 51593
"""
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
from scipy import *
from scipy.integrate import simps, cumtrapz, romb
# matplotlib inline
import math
#import ql_exact
#Generalized Marsaglia
def GMHeston(kappa, theta, beta, rho, v0 ,r ,T ,s0 ,K,N,dt):
    start_time = time.time()
    Ntime=int(T/dt)
    sqrt_dt=sqrt(dt)
    S=np.ones(N)*s0
    v=np.ones(N)*v0
    vega=(4*kappa*theta/(beta*beta))
    K1=dt*(kappa*rho/beta-0.5)/2-rho/beta
    K2=dt*(kappa*rho/beta-0.5)/2+rho/beta
    K3=dt*(1-rho*rho)/2
    ss=K2+K3/2
    yita=4*kappa*exp(-kappa*dt)/(beta*beta)/(1-exp(-kappa*dt))
    sh=ss*exp(-kappa*dt)/yita
    for i in range (0,Ntime):   
        Zs=np.random.randn(1,N)
        lamb=v*yita
        W=np.random.noncentral_chisquare(vega,lamb)
        v2=W*exp(-kappa*dt)/yita
        K0=-lamb*sh/(1-2*sh)+0.5*vega*log(1-2*sh)-(K1+K3/2)*v
        vreal=np.real(np.maximum(v,0))
        S=S*exp(r*dt+K0+K1*v+K2*v2+np.multiply(sqrt(K3*(v+v2)),Zs))
        v=v2
    payoff=np.maximum(S-K,0)*exp(-r*T)
    std=np.std(payoff)/sqrt(N)
    price=np.mean(payoff)
    TIME=(time.time() - start_time)
    Result = collections.namedtuple('Result', ['price', 'std','time'])
    out=Result(price,std=std,time=TIME)      
    return (out)
    
if __name__ == '__main__':
    T = 5
    #risk free rate
    r = 0.05
    #long term volatility(equiribrium level)
    theta = 0.09
    #Mean reversion speed of volatility
    kappa = 1
    #beta(volatility of Volatility)
    beta = 1
    #rho
    rho = -0.3
    #Initial stock price
    s0 = 100.0
    #Initial volatility
    v0 = 0.09
    #0.634
    K = np.arange(60, 180, 40)
    for i in range(0,3):
        print (GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),100000,1/16))
        #convert int32 to int
     
