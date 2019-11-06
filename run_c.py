import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix, show_heatmaps
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from functions import discretize, make_onehot, inverse_onehot
from neural_network import NeuralNetwork, layer
import seaborn as sns

#k-fold cross validation parameters
CV = False
k = 5

#random dataset, or credit card?
randomdataset = False

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 1000
n_samples = 100000      #Only for generation of random dataset


if randomdataset:
    #generate random dataset
    X, y = datasets.make_classification(n_samples=n_samples, n_features=20,
                                        n_informative=2, n_redundant=2)
    CDds = dataset(0)
    val = np.column_stack((X,y))
    CDds.values = val
    
    #polishing the dataset, and divide into data and target data
    CDds.polish_and_divide()
else:
    #Importing the credit card dataset
    filename = "credit_card_data_set.xls"
    CDds = credit_card_dataset(filename)
    
    #polishing the dataset, and divide into data and target data
    CDds.CreditCardPolish()
    
#Normalize dataset
CDds.normalize_dataset()

#Divide in train and test
CDds.sort_train_test(ratio = 0.2, random = False)

#Initialize inputs for Neural Network
X_train = CDds.x_1d
y_train = CDds.y_1d[:,np.newaxis]
y_train_onehot = make_onehot(y_train)
if not randomdataset:
    n_samples = X_train.shape[0]




###### grid search #######

#Initialize vectors for saving values
#eta_vals = np.logspace(-6, -1, 6)
eta_vals = np.linspace(1e-5, 1e-3, 7)
lmbd_vals = np.logspace(-5, 1, 7)
FFNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
train_rocauc = np.zeros((len(eta_vals), len(lmbd_vals)))
test_rocauc = np.zeros((len(eta_vals), len(lmbd_vals)))

#Loop through the etas and lambdas
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #Make neural network
        ffnn = NeuralNetwork(X_train, 
                             y_train_onehot, 
                             batch_size=int(n_samples/m), 
                             n_categories = 2, 
                             epochs = 100, 
                             n_hidden_neurons = 50, 
                             eta = eta,
                             lmbd = lmbd)
        ffnn.add_layer(50)
        
        #Train network
        ffnn.train()
        
        #Save predictions
        y_tilde_train = ffnn.predict_probabilities(X_train)
        y_tilde_train_1d = ffnn.predict(X_train)
        y_tilde = ffnn.predict_probabilities(CDds.test_x_1d)
        y_tilde_1d = ffnn.predict(CDds.test_x_1d)
        
        #target for the test set
        _, target = CDds.rescale_back(x = CDds.test_x_1d, y = CDds.test_y_1d, split = True)
        target = [int(elem) for elem in target]
        
        #Save prediction into exportable matrices
        train_accuracy[i][j] = statistics.calc_accuracy(y_train, y_tilde_train_1d)
        test_accuracy[i][j] = statistics.calc_accuracy(target, y_tilde_1d)
        train_rocauc[i][j] = roc_auc_score(y_train_onehot, y_tilde_train)
        test_rocauc[i][j] = roc_auc_score(make_onehot(target), y_tilde)
        
        #print some outputs
        print('Learning rate: ', eta)
        print('lambda: ', lmbd)
        print('rocauc: ', test_rocauc[i][j])
        print('accuracy: ', test_accuracy[i][j])
        print('\n')
            
            
#Visualization        
show_heatmaps(lmbd_vals, eta_vals, train_accuracy, test_accuracy, train_rocauc, test_rocauc)

