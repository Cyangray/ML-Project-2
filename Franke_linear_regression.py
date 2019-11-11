import numpy as np
from dataset_objects import dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling

#k-fold cross validation parameters
CV = True
k = 5

# Regression Parameters
method = 'OLS'
lambd = 0.01

#Degree 0 because it's a classification and not a polynomial
deg = 5

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 10e5 #also valid as maxiter for LASSO

#Get random seed
np.random.seed(1234)

#generate random dataset of the Franke function with noise
FrankeDS = dataset(0)
FrankeDS.generate_franke(150, 0.05)
    
#Normalize dataset
FrankeDS.normalize_dataset()

#Divide in train and test
if CV:
    FrankeDS.sort_in_k_batches(k)
else:
    FrankeDS.sort_train_test(ratio = 0.2, random = False)

#Make model
FrankeModel = fit(FrankeDS)

#Create polynomial design matrix for train and test sets
X_train = FrankeModel.create_design_matrix(deg = deg)

if CV:
    # Run k-fold CV algorithm and fit models.
    sample = sampling(FrankeDS)
    sample.kfold_cross_validation(method, deg=deg, lambd = lambd, Niterations = Niterations)
    
    # Print metrics
    print("Cross-validation batches: k = ", k)
    print('Best train mse is in arg ', np.argmin(sample.mse_train), ' : ', min(sample.mse_train))
    print('Best train R2 score is in arg ', np.argmax(sample.R2_train),' : ', max(sample.R2_train))
    print('Best test mse is in arg ', np.argmin(sample.mse), ' : ', min(sample.mse))
    print('Best test R2 score is in arg ', np.argmax(sample.R2),' : ', max(sample.R2))
    print('test mses: ', sample.mse)
    print('test R2s: ', sample.R2)

    
else:
    #Dont run k-fold CV
    #collect information about training set
    if method == 'OLS':
        y_tilde_train, betas = FrankeModel.fit_design_matrix_numpy()
    elif method == 'Ridge':
        y_tilde_train, betas = FrankeModel.fit_design_matrix_ridge(lambd = lambd)
    elif method == 'LASSO':
        y_tilde_train, betas = FrankeModel.fit_design_matrix_lasso(lambd = lambd, maxiter = Niterations)
    target_train = FrankeDS.y_1d
    #_, target_train = FrankeDS.rescale_back(x = FrankeDS.x_1d, y = FrankeDS.y_1d, split = True)
    
    #collect information about test set
    X_test = FrankeModel.create_design_matrix(x = FrankeDS.test_x_1d, deg = 5)
    y_tilde = FrankeModel.test_design_matrix(betas, X = X_test)
    target = FrankeDS.test_y_1d
    #_, target = FrankeDS.rescale_back(x = FrankeDS.test_x_1d, y = FrankeDS.test_y_1d, split = True)
    #_, y_tilde_scaled = FrankeDS.rescale_back(x = FrankeDS.test_x_1d, y = y_tilde, split = True)
    
    mse_train = statistics.calc_MSE(target_train, y_tilde_train)
    R2_train = statistics.calc_R2_score(target_train, y_tilde_train)
    mse_test = statistics.calc_MSE(target, y_tilde)
    R2_test = statistics.calc_R2_score(target, y_tilde)
    
    #print metrics
    print('MSE train = ', mse_train)
    print('R2 train = ', R2_train)
    print('MSE test = ', mse_test)
    print('R2 test = ', R2_test)

