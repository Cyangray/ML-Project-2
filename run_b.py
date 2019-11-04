import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from functions import discretize


#k-fold cross validation parameters
CV = False
randomdataset = True
k = 5
method = 'logreg'
deg = 0

#Stochastic gradient descent parameters
m = 5           #Number of minibatches
Niterations = 1000



if randomdataset:
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=2)
    CDds = dataset(0)
    val = np.column_stack((X,y))
    dataset.values = val
    liste = [CDds]
    CDds.polish_and_divide()
else:
    #Importing the credit card dataset
    filename = "credit_card_data_set.xls"
    CDds = credit_card_dataset(filename)
    liste = [CDds]
    
    #polishing the dataset, and divide into data and target data
    CDds.CreditCardPolish()
    
#Normalize dataset
CDds.normalize_dataset()

#Divide in train and test
if CV:
    CDds.sort_in_k_batches(k)
else:
    CDds.sort_train_test(ratio = 0.2)

#Make model
model = fit(CDds)

#Fit model
model.create_simple_design_matrix()

if CV:
    # Run k-fold CV algorithm and fit models.
    sample = sampling(CDds)
    liste2 = [sample]
    sample.kfold_cross_validation(method, deg=deg, descent_method = 'skl-SGD', Niterations = Niterations, m = m)
    # Calculate statistics and write output
    print('Number of epochs: ', int(Niterations/m))
    print("Cross-validation batches: k = ", k)
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)
    
    #print accuracy
    print('accuracy is: ', sample.accuracy)
    print('roc-auc score is: ', sample.rocaucs)
    
else:
    #Dont run k-fold CV
    
    y_tilde_train, betas = model.fit_design_matrix_logistic_regression(descent_method = 'skl-SGD', Niteration = Niterations, m = m)
    _, target_train = CDds.rescale_back(x = CDds.x_1d, y = CDds.y_1d, split = True)
    
    X_test = model.create_design_matrix(x = CDds.test_x_1d)
    y_tilde = model.test_design_matrix(betas, X = X_test)
    _, target = CDds.rescale_back(x = CDds.test_x_1d, y = CDds.test_y_1d, split = True)
    
    print('Number of epochs: ', int(Niterations/m))
    print('Training set accuracy is: ', statistics.calc_accuracy(target_train, y_tilde_train))
    print('Test set accuracy is: ', statistics.calc_accuracy(target, y_tilde))
    train_rocauc = roc_auc_score(target_train, y_tilde_train)
    test_rocauc = roc_auc_score(target, y_tilde)
    print('Training roc-auc score is: ', train_rocauc)
    print('Test roc-auc score is: ', test_rocauc)









    
    
    
    
    
    
    