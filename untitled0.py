#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:08:54 2022

@author: Teresa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


numberOfSamples = 30

mu, sigma = 0, 1 # mean and standard deviation
noise = np.random.normal(mu, sigma, numberOfSamples)

X = np.random.rand(numberOfSamples)*5
Y = 2*X + 1 + noise
 
plt.axis([0, 5, 0, 12])
plt.plot(X,Y,'bo')
plt.show()


def dm(X,y,m,b):
    dervm = np.sum(2*(- y + m*X+b)*X)
    return dervm
    
def db(X,y,m,b):
    dervb = np.sum(2*(m*X +b - y))
    return dervb

def update(param,devparam,l):
    return param - l*devparam

l = 0.5  # esto creo que es lambda el parámetro de aprendizaje, no se con que criterio se elige
ite = 100













