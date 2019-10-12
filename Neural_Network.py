#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:02:40 2019

@author: kaju
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def plotSigmoid(range_val):
    x = np.linspace(0, range_val)
    y = sigmoid(x)
    plt.plot(x,y)
    
#plotSigmoid(10)
#plt.show()
    
