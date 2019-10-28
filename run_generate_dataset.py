import numpy as np 
import sys
import os


from data_generation import data_generate

"""Generates a data set which can be loaded in other functions for statistical consistency"""


n = 150                 # no. of x and y coordinates
deg = 5                 # degree of polynomial
noise = 0.05            # if zero, no contribution. Otherwise scaling the noise.

# Load dataset and generate Franke function
dataset = data_generate()
dataset.generate_franke(n, noise)
dataset.save_data()