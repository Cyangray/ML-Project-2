import numpy as np
from functions import make_onehot, softmax, sigmoid
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import datasets


#k-fold cross validation parameters
CV = False
k = 5

# Regression Parameters
method = 'logreg'

#Change this to 'GD' to obtain the Gradient descent results, to 'SGD' to obtain
#the Stochastic gradient descent result, or to 'skl-SGD' to use the scikit-learn algorithm
desc_method = 'SGD' 

#This is eta0, or learning rate.
input_eta = 1.

#Degree 0 because it's a classification and not a polynomial
deg = 0


#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 5000


#Random dataset or Credit card?
randomdataset = False

#put random seed
np.random.seed(1234)


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
    CDds.sort_train_test(ratio = 0.2, random = False)

#Make model
model = fit(CDds)

#Fit model
model.create_simple_design_matrix()

if CV:
    # Run k-fold CV algorithm and fit models.
    sample = sampling(CDds)
    liste2 = [sample]
    sample.kfold_cross_validation(method, deg=deg, descent_method = desc_method, eta = input_eta, Niterations = Niterations, m = m)
    
    # Print metrics
    print('Number of epochs: ', int(Niterations/m))
    print("Cross-validation batches: k = ", k)
    print('Best accuracy is in arg ', np.argmax(sample.accuracy), ' : ', max(sample.accuracy))
    print('Best roc-auc score is in arg ', np.argmax(sample.rocaucs),' : ', max(sample.rocaucs))
    print('Best area ratio is in arg ', np.argmax(sample.area_ratios),' : ', max(sample.area_ratios), '\n')
    print('accuracy is: ', sample.accuracy)
    print('roc-auc score is: ', sample.rocaucs)
    print('Area ratio is: ', sample.area_ratios, '\n')
    
else:
    #Dont run k-fold CV
    #collect information about training set
    y_tilde_train, betas = model.fit_design_matrix_logistic_regression(descent_method = desc_method, eta = input_eta, Niteration = Niterations, m = m)
    _, target_train = CDds.rescale_back(x = CDds.x_1d, y = CDds.y_1d, split = True)
    target_train = [int(elem) for elem in target_train]
    
    #collect information about test set
    X_test = model.create_design_matrix(x = CDds.test_x_1d)
    y_tilde = sigmoid(model.test_design_matrix(betas, X = X_test))
    _, target = CDds.rescale_back(x = CDds.test_x_1d, y = CDds.test_y_1d, split = True)
    _, y_tilde_scaled = CDds.rescale_back(x = CDds.test_x_1d, y = y_tilde, split = True)
    target = [int(elem) for elem in target]
    
    #Make onehot version of results
    y_tilde_train_onehot = np.column_stack((1 - y_tilde_train, y_tilde_train))
    y_tilde_onehot = np.column_stack((1 - y_tilde, y_tilde))
    
    # Print metrics
    print('Number of epochs: ', int(Niterations/m))
    print('Training set accuracy is: ', accuracy_score(target_train, np.argmax(y_tilde_train_onehot, axis = 1)))
    print('Test set accuracy is: ', accuracy_score(target, np.argmax(y_tilde_onehot, axis = 1)))
    print('Training roc-auc score is: ', roc_auc_score(target_train, y_tilde_train))
    print('Test roc-auc score is: ', roc_auc_score(target, y_tilde))
    
    max_area_ratio_train = statistics.calc_cumulative_auc(target_train, make_onehot(target_train))
    max_area_ratio_test = statistics.calc_cumulative_auc(target, make_onehot(target))
    print('Training area ratio is: ', (statistics.calc_cumulative_auc(target_train, y_tilde_train_onehot) - 0.5)/(max_area_ratio_train - 0.5))
    print('Test area ratio is: ', (statistics.calc_cumulative_auc(target, y_tilde_onehot) - 0.5)/(max_area_ratio_test - 0.5))









    
    
    
    
    
    
    