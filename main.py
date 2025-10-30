# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:41:11 2018

@author: rsubrama
This code was written using the course material of Dr. Andrew Ng for my statistical course project.
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from Forward_Propagation import L_model_forward
from Back_Propagation import L_model_backward
from update_parameters import update_parameters
from prediction import predict

from Initialization_Hyperparameters import initialize_hyperparameters_deep
from CostFunction import compute_mse


def Deep_model(X, Y, layers_dims, learning_rate = 0.009, num_iterations = 3000, print_mse=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3) ------- > (800,23)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)  (1,800)
    """

    np.random.seed(1)
    Training_MSE = []
    
    # Parameters initialization. 
    parameters = initialize_hyperparameters_deep(layers_dims)    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)     
        # Compute MSE.
        assert(AL.shape == (1,800))
        assert(Y.shape == (1,800))
        mse = compute_mse(AL, Y)    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)     
    
    return parameters
 
def main():
    #Initializations
    """ Adjust your neural network layers here"""
    layers_dims = [23,200,200,200,200,200,1]
    #Reading Input Data
    cancer_data = pd.read_csv("C:\\Users\\rsubrama\\Documents\\Courses\\Fall-2018\\Intro to Statistical Learning\\cancer-patient-data_deep.csv",header=None)
    print(type(cancer_data))
    #Reading Output Label
    cancer_data_out = pd.read_csv("C:\\Users\\rsubrama\\Documents\\Courses\\Fall-2018\\Intro to Statistical Learning\\cancer-patient-data_deep_output.csv",header=None)
    #Normalizing the data from 0 to 1
    print(cancer_data[0])
    cancer_data[0] = cancer_data[0]/100
    cancer_data[1::] = cancer_data[1::]/10
    #Changing into numpy arrays
    cancer_data = cancer_data.values
    cancer_data_out = cancer_data_out.values
    #Splitting the dataset for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(cancer_data, cancer_data_out , test_size=0.2, random_state = 74)
    #X_train is (23,800) and Y_train is (1,800)
    #Transposing the training and testing sets for learning
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    Y_train = Y_train.transpose()
    Y_test = Y_test.transpose()     
    #Learnt_parameters = Deep_model(X_train, Y_train, layers_dims, num_iterations = 2500, print_mse = True)      
    #Prediction
    #Accuracy = predict(X_train, Y_train, Learnt_parameters)
    #print("Accuracy of the model:"+ Accuracy + "%")
    

if __name__ == '__main__':

    main()
