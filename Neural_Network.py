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
        
    def calcGrad(self):
        
        """
        Activation function
        calculating the derivative wrt x. f'(x) = f(x)(1-f(x))
        """
        grad_mat = np.dot(self.output, (1 - self.output).T)
        grad_activate = np.diag(np.diag(grad_mat))
        return grad_activate
    
    def fwdPropagation(self, fx_dat):
        
        self.fx_dat = fx_dat
        if self.check_input_layer:
            self.weight_x_plus_bias = fx_dat
            self.output = fx_dat
        else:
            self.weight_x_plus_bias = (np.dot(self.weight, self.fx_dat) 
                                        - self.bias)
            self.output = self.activation(self.weight_x_plus_bias)
            return self.output
        
    def backPropagation(self, back_grad):
        
        back_grad_activation = self.calcGrad()
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
    
class NeuralModel:
    
    def __init__(self):
        self.layers = []
        self.training_mse = []
        self.plot_loss = plt.figure()
        self.plot_ax_loss = self.plot_loss.add_subplot(1,1,1)
        
    def layerAddition(self, layer):
        self.layers.append(layer)
        
    def buildNetwork(self):
        for i, layer in enumerate(self.layers[:]):
            if i < 1:
                layer.check_input_layer = True
            else:
                layer.initialize(self.layers[i-1].units)
                
    def train(self, fx_dat, y_dat, train_round, accuracy):
        self.train_round = train_round
        self.accuracy = accuracy
        self.plot_ax_loss.hlines(self.accuracy, 0, self.train_round * 1.11)
        
        x_shape = np.shape(fx_dat)
        for train_round_no in (train_round):
            sum_loss = 0
            
            for row in range(x_shape[0]):
                x_train_data = np.asmatrix(fx_dat[row, :]).T
                y_train_data = np.asmatrix(y_dat[row, :]).T
                
                for layer in self.layers:
                    x_train_data = layer.fwdPropagation(x_train_data)
                    
                loss, gradient = self.calLoss(y_train_data, x_train_data)
                sum_loss = sum_loss + loss
                
                for layer in self.layers[:0:-1]:
                    gradient = layer.backPropagation(gradient)
                    
                
                mse = sum_loss / x_shape[0]
                self.training_mse.append(mse)
                
                self.plotBackPropLoss()
                
                if mse < self.accuracy:
                    return mse
                
                
    def calLoss(self, orig_y_data, calc_y_data):
        
        self.loss = np.sum(np.power((orig_y_data - calc_y_data), 2))
        self.loss_gradient = 2 * (orig_y_data - calc_y_data)
        
        return self.loss, self.loss_gradient
    
    def plotBackPropLoss(self):
        
        if self.plot_ax_loss.lines:
            self.plot_ax_loss.remove(self.plot_ax_loss[0])
            
        self.plot_ax_loss(self.training_mse, "b+")
        plt.ion()
        plt.xlabel("train_round")
        plt.ylabel("y_loss")
        plt.show()
        plt.pause(0.2)
        

def testMatrix():
    x = np.random.randn(10, 10)
    y = np.asarray(
        [
            [0.2, 0.3],
            [0.5, 0.9],
            [0.47, 0.41],
            [0.77, 0.57],
            [0.23, 0.96],
            [0.74, 0.12],
            [0.14, 0.47],
            [0.96, 0.23],
            [0.21, 0.15],
            [0.5, 0.1],
        ]
    )
    
    model = NeuralModel()
    model.layerAddition(ConnectedLayer(15))
    model.layerAddition(ConnectedLayer(20))
    model.layerAddition(ConnectedLayer(30))
    model.layerAddition(ConnectedLayer(2)) # output layer
    
    model.buildNetwork()
    
    model.train(fx_dat=x, y_dat=y, train_round=50, accuracy=0.02)
    
if __name__== "__main__":
    testMatrix()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        