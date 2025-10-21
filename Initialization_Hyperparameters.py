# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:12:27 2018

@author: rsubrama
"""
import pandas as pd
import numpy as np

def initialize_hyperparameters_deep(Network_dimension):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(Network_dimension)            # number of layers in the network

    for l in range(1, L):
        #He Initialization
        parameters['W' + str(l)] = np.random.randn(Network_dimension[l], Network_dimension[l-1]) * np.sqrt(np.divide(2,Network_dimension[l-1]))
        parameters['b' + str(l)] = np.zeros((Network_dimension[l], 1))
        
        assert(parameters['W' + str(l)].shape == (Network_dimension[l], Network_dimension[l-1]))
        assert(parameters['b' + str(l)].shape == (Network_dimension[l], 1))
        
    return parameters