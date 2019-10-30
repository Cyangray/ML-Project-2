import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling

#k-fold cross validation parameters
k = 5
method = 'logreg'
deg = 0

#Stochastic gradient descent parameters
m = 5           #Number of minibatches

#Importing the credit card dataset
filename = "credit_card_data_set.xls"
CDds = credit_card_dataset(filename)
liste = [CDds]

#polishing the dataset, and divide into data and target data
CDds.CreditCardPolish()

#Normalize dataset
CDds.normalize_dataset()

#Divide in train and test
CDds.sort_in_k_batches(k)

#Make model
#model = fit(CDds)

#Fit model
#model.create_simple_design_matrix()



# Run k-fold algorithm and fit models.
sample = sampling(CDds)
liste2 = [sample]
sample.kfold_cross_validation(method, deg=deg, descent_method = 'SGD', m = m)

# Calculate statistics
print("Batches: k = ", k)
statistics.print_mse(sample.mse)
statistics.print_R2(sample.R2)


#y_tilde, betas = model.fit_design_matrix_logistic_regression(descent_method = 'SGD-skl', verbose = True)

#print accuracy
print('accuracy is: ', sample.accuracy)


    
    
    
    
    
    
    