# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:00:56 2018

@author: 51593
"""

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
from scipy import *
from scipy.integrate import simps, cumtrapz, romb
# matplotlib inline
import math
def call_price_exact(kappa, theta, beta, rho, v0 ,r ,T ,s0 ,K):
    strike_price = 110.0
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
    # option data
    calculation_date = ql.Date(15, 1, 2011)
    maturity_date = ql.Date(15, 1, 2011+T)
    spot_price = s0
    strike_price = K
    dividend_rate =  0.00
    option_type = ql.Option.Call   
    risk_free_rate = r
    day_count = ql.Actual365Fixed()    
    ql.Settings.instance().evaluationDate = calculation_date
    # construct the Heston process
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)
    sigma = beta   
    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, dividend_rate, day_count)
    )
    heston_process = ql.HestonProcess(flat_ts,
                                      dividend_yield,
                                      spot_handle,
                                      v0,
                                      kappa,
                                      theta,
                                      sigma,
                                      rho)
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.0000001, 100000)
    european_option.setPricingEngine(engine)
    h_price = european_option.NPV()
    return h_price
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import black_sholes
    #maturity
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
        print (call_price_exact(kappa, theta, beta, rho, v0, r, T, s0, K[i].item()))
        #convert int32 to int
     
    #Strikes
    K = np.arange(60, 180, 40)
    #simulation
    imp_vol = np.array([])
    for k in K:
        #calc option price
        price = call_price_exact(kappa, theta, beta, rho, v0, r, T, s0, k.item())
        #calc implied volatility
        imp_vol = np.append(imp_vol, black_sholes.implied_volatility(price, s0, k, T, r, 'C'))
        #print (k, price, imp_vol[-1])

    #plot result
