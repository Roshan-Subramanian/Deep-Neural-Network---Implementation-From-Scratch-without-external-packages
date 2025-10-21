# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:37:42 2018

@author: rsubrama
"""
import numpy as np

def compute_mse(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    mse = (np.square(AL - Y)).mean(axis=None)  
    mse = np.squeeze(mse)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(mse.shape == ())    
    return mse