# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:20:06 2018

@author: rsubrama
"""
import numpy as np
from Forward_Propagation import L_model_forward

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probability, caches = L_model_forward(X, parameters)
    #print(probability)
    
    #convert probas to 0/1 predictions
    for i in range(0, probability.shape[1]):
        if probability[0,i] > 0.7:
            probability[0,i] = 1    
               
        elif probability[0,i] > 0.4 and probability[0,i] < 0.7:
            probability[0,i] = 0.5
        else:
            probability[0,i] = 0
        
    return str(np.sum((probability == y)/m)*100)