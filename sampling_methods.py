import numpy as np 

import statistical_functions as statistics
from fit_matrix import fit
from functions import franke_function
import copy

class sampling():
    def __init__(self, inst):
        self.inst = inst

    def kfold_cross_validation(self, method, descent_method='SGD-skl', deg=0, lambd=0.01):
        """Method that implements the k-fold cross-validation algorithm. It takes
        as input the method we want to use. if "least squares" an ordinary OLS will be evaulated.
        if "ridge" then the ridge method will be used, and respectively the same for "lasso"."""

        inst = self.inst
        lowest_mse = 1e5

        self.mse = []
        self.R2 = []
        self.mse_train = []
        self.R2_train = []
        self.bias = []
        self.variance = []
        self.accuracy = []
        self.design_matrix = fit(inst)
        #whole_DM = self.design_matrix.create_design_matrix(deg=deg).copy() #design matrix for the whole dataset
        #whole_y = inst.y_1d.copy() #save the whole output
        
        for i in range(self.inst.k):
            #pick the i-th set as test
            inst.sort_training_test_kfold(i)
            inst.fill_array_test_training()
            self.design_matrix.create_design_matrix(deg = deg) #create design matrix for the training set, and evaluate
            
            if method == 'OLS':
                y_train, beta_train = self.design_matrix.fit_design_matrix_numpy()
            elif method == "ridge":
                y_train, beta_train = self.design_matrix.fit_design_matrix_ridge(lambd)
            elif method == "lasso":
                y_train, beta_train = self.design_matrix.fit_design_matrix_lasso(lambd)
            elif method == 'logreg':
                y_train, beta_train = self.design_matrix.fit_design_matrix_logistic_regression(descent_method = descent_method)
                
            else:
                sys.exit("Wrongly designated method: ", method, " not found")

            #Find out which values get predicted by the training set
            X_test = self.design_matrix.create_design_matrix(x=inst.test_x_1d, N=inst.N_testing, deg=deg)
            y_pred = self.design_matrix.test_design_matrix(beta_train, X=X_test)

            #Take the real target values from the test datset for comparison (and also a rescaled set)
            y_test = inst.test_y_1d
            transformed_matrix = inst.rescale_back(x = inst.test_x_1d, y = inst.test_y_1d)
            y_test_rescaled = transformed_matrix[:,-1]
            self.y_test_rescaled = y_test_rescaled
            self.y_test_rescaled_int = y_test_rescaled.astype(int)
            
            #Calculate the prediction for the whole dataset
            #whole_y_pred = self.design_matrix.test_design_matrix(beta_train, X=whole_DM)

            # Statistically evaluate the training set with test and predicted solution.
            mse, calc_r2 = statistics.calc_statistics(y_test, y_pred)
            
            # Statistically evaluate the training set with itself
            mse_train, calc_r2_train = statistics.calc_statistics(inst.y_1d, y_train)
            
            # Get the values for the bias and the variance
            bias, variance = statistics.calc_bias_variance(y_test, y_pred)
            
            accuracy_batch = statistics.accuracy(self.y_test_rescaled_int, y_pred)
            
            self.mse.append(mse)
            self.R2.append(calc_r2)
            self.mse_train.append(mse_train)
            self.R2_train.append(calc_r2_train)
            self.bias.append(bias)
            self.variance.append(variance)
            self.accuracy.append(accuracy_batch)
            # If needed/wanted: 
            if abs(mse) < lowest_mse:
                lowest_mse = abs(mse)
                self.best_predicting_beta = beta_train
            
