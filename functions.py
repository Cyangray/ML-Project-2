import numpy as np
from math import floor

def discretize(a):
    discretized_a = np.zeros(np.shape(a))
    for i, element in enumerate(a):
        if element > 0:
            discretized_a[i] = 1
    return discretized_a

def make_onehot(a):
    uniques = np.unique(a)
    n_classes = len(uniques)
    a_onehot = np.zeros((len(a),n_classes))
    for i, elem in enumerate(a):
        for j, unique in enumerate(uniques):
            if elem == unique:
                a_onehot[i,j] = 1
    return a_onehot

def inverse_onehot(a_onehot):
    a = np.zeros(a_onehot.shape[0])
    for i in range(len(a)):
        a[i] = np.argmax(a_onehot[i,:])
    return a

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def franke_function(x,y):
    """ Generate values for the franke function"""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def is_odd(num):
    """ Returns True if number is odd, False is even. Used for scaling data. """
    return num & 0x1

def reduce4(A):
    """ Reduce the dimension of a matrix by four times, by only taking the first 
    value of every second for both axis"""
    
    A_rows = np.size(A,0)
    A_columns = np.size(A,1)
    A_rows_list = range(A_rows)
    A_columns_list = range(A_columns)
    
    B_rows = floor(A_rows/2)
    B_columns = floor(A_columns/2)
    
    B = np.zeros((B_rows, B_columns ))
    
    AtoB_rows = [A_rows_list[i]*2 for i in range(B_rows)]
    AtoB_columns = [A_columns_list[i]*2 for i in range(B_columns)]
    
    for i, row in enumerate(AtoB_rows):
        B[i,:] = A[row, AtoB_columns] 
    
    return B