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
def KJHeston(kappa, theta, beta, rho, v0 ,r ,T ,s0 ,K,N,dt):
    start_time = time.time()
    Ntime=int(T/dt)
    sqrt_dt=sqrt(dt)
    S=np.ones(N)*log(s0)
    v=np.ones(N)*v0
    for i in range (0,Ntime):   
        Zv=np.random.randn(1,N)
        Ztemp=np.random.randn(1,N)
        Zs=rho*Zv+sqrt(1-(rho*rho))*Ztemp   
        v2=(v+kappa*theta*dt+beta*np.multiply(sqrt(v),Zv)*sqrt_dt+1/4*beta*beta*dt*(np.power(Zv,2)-1))/(1+kappa*dt)
        v2=v2[0,:]
        idx=np.where(np.iscomplex(v))[0]
        vreal=np.real(np.maximum(v,0)) 
        v2[idx]=v[idx]+kappa*dt*(theta-vreal[idx])+beta*np.multiply(sqrt(vreal[idx]),Zv[0,idx])*sqrt_dt
        v2real=np.real(np.maximum(v2,0))
        v2=np.real(v2)
        #here S means log S
        S=S+(r-(vreal+v2real)/4)*dt+rho*np.multiply(sqrt(vreal),Zv)*sqrt_dt+ \
        0.5*np.multiply((sqrt(vreal)+sqrt(v2real)),(Zs-rho*Zv))*sqrt_dt+((rho*beta*dt)/4)* \
        (np.multiply(Zv,Zv)-1)
        v=v2
        S=np.real(np.maximum(S,0))
    payoff=np.maximum(exp(S)-K,0)*exp(-r*T)
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
        print (KJHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),100000,1/16).std)
        #convert int32 to int
     