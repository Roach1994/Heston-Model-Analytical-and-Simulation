# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:39:39 2018

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
def EulerHeston(kappa, theta, beta, rho, v0 ,r ,T ,s0 ,K,N,dt):
    start_time = time.time()
    Ntime=int(T/dt)
    sqrt_dt=sqrt(dt)
    S=np.ones(N)*log(s0)
    v=np.ones(N)*v0
    for i in range (0,Ntime):   
        Zv=np.random.randn(1,N)
        Ztemp=np.random.randn(1,N)
        Zs=rho*Zv+sqrt(1-(rho*rho))*Ztemp
        #S=S*(1+r*dt+sqrt(v)*Zs*sqrt_dt)
        vreal=np.real(np.maximum(v,0))
        S=S-0.5*vreal*dt+np.multiply(sqrt(vreal),Zs)*sqrt_dt+r*dt
        #S=np.multiply((1+r*dt+np.multiply(sqrt(vreal),Zs)*sqrt_dt),S)
        #v=v+kappa*dt*(theta-v)+beta*sqrt(v)*Zv*sqrt_dt
        v=v+kappa*dt*(theta-vreal)+beta*np.multiply(sqrt(vreal),Zv)*sqrt_dt
    payoff=np.maximum(exp(S)-K,0)*exp(-r*T)
    std=np.std(payoff)/sqrt(N)
    price=np.mean(payoff)
    TIME=(time.time() - start_time)
    Result = collections.namedtuple('Result', ['price', 'std','time'])
    out=Result(price,std=std,time=TIME)      
    return (out)
    
if __name__ == '__main__':
    #import timeit
    #print(timeit.timeit("test()", setup="from __main__ import test"))
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
        print (EulerHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),100000,1/32))
        #convert int32 to int
     