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
    
class ConnectedLayer:
    
    """
    Neural Layers
    """
    
    def __init__(self, units, activation=None, learning_rate=None, 
                 check_input_layer=False):
        
        self.units = units
        self.weight = None
        self.bias = None
        self.activation = activation
        
        if learning_rate is None:
            learning_rate = 0.29
        self.learning_rate = learning_rate
        
        self.check_input_layer = check_input_layer
        
    def initialize(self, back_prop_units):
        self.weight = np.asmatrix(np.random.normal(0, 0.5, (self.units, 
                                                            back_prop_units)))
        
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.units)).T
        self.activation = sigmoid
        
    def calc_grad(self):
        
        """
        Activation function
        calculating the derivative wrt x. f'(x) = f(x)(1-f(x))
        """
        grad_mat = np.dot(self.output, (1 - self.output).T)
        grad_activate = np.diag(np.diag(grad_mat))
        return grad_activate
    
    def fwd_propagation(self, fx_dat):
        
        self.fx_dat = fx_dat
        if self.check_input_layer:
            self.weight_x_plus_bias = fx_dat
            self.output = fx_dat
        else:
            self.weight_x_plus_bias = (np.dot(self.weight, self.fx_dat) 
                                        - self.bias)
            self.output = self.activation(self.weight_x_plus_bias)
            return self.output
        
    def back_propagation(self, back_grad):
        
        back_grad_activation = self.calc_grad()
        back_grad = np.asmatrix(np.dot(back_grad.T, back_grad_activation))
        
        self.back_grad_weight = np.asmatrix(self.fx_dat)
        self.back_grad_bias = -1
        self.back_grad_x = self.weight
        
        self.grad_weight = np.dot(back_grad.T, self.back_grad_weight.T)
        self.grad_bias = back_grad * self.back_grad_bias
        self.back_grad = np.dot(back_grad, self.back_grad_x).T
        self.weight = self.weight - self.learning_rate * self.grad_weight
        self.bias = self.bias - self.learning_rate  * self.grad_bias.T
        
        return self.back_grad
    


        
        