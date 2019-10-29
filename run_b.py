#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:07:20 2019

@author: francesco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit

#Importing the credit card dataset
filename = "credit_card_data_set.xls"
CDds = credit_card_dataset(filename)
liste = [CDds]

#polishing the dataset, and divide into data and target data
CDds.CreditCardPolish()

#Normalize dataset
CDds.normalize_dataset()

#Make model
model = fit(CDds)

#Fit model
model.create_simple_design_matrix()
y_tilde, betas = model.fit_design_matrix_logistic_regression(method = 'a')



y10 = np.zeros(np.shape(y_tilde))
right_guesses = 0
for i, yi in enumerate(y_tilde):
    if yi > 0:
        y10[i] = 1
    else:
        y10[i] = 0
    
    if y10[i] == CDds.y_1d_unscaled[i]:
        right_guesses += 1
accuracy = right_guesses / len(CDds.y_1d)
    
    
    
    
    
    
    