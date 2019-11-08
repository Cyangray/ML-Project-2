import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix, show_heatmap_mse_R2, plot_3d
from dataset_objects import dataset, credit_card_dataset
from fit_matrix import fit
import statistical_functions as statistics
from sampling_methods import sampling
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from functions import make_onehot, inverse_onehot
from neural_network import NeuralNetwork, layer
import seaborn as sns

#k-fold cross validation parameters
CV = False
k = 5
np.random.seed(0)

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 1000
deg = 5


#generate random dataset of the Franke function with noise
FrankeDS = dataset(0)
FrankeDS.generate_franke(150, 0.05)

#polishing the dataset, and divide into data and target data
FrankeDS.polish_and_divide()
    
#Normalize dataset
FrankeDS.normalize_dataset()

#Divide in train and test
FrankeDS.sort_train_test(ratio = 0.2, random = False)

#Make model
FrankeModel = fit(FrankeDS)

#Create polynomial design matrix for train and test sets
X_train = FrankeModel.create_design_matrix(deg = deg)
X_test = FrankeModel.create_design_matrix(x = FrankeDS.test_x_1d, deg = deg)

#Initialize inputs for Neural Network
y_train = FrankeDS.y_1d[:,np.newaxis]
y_test = FrankeDS.test_y_1d[:,np.newaxis]
n_samples = X_train.shape[0]

###### grid search #######

#Initialize vectors for saving values
eta_vals = np.logspace(-3, 0, 10)
#eta_vals = np.linspace(1e-6, 1e-3, 10)
lmbd_vals = np.logspace(-6, 1, 8)
FFNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
train_R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
test_R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
best_train_mse = 10.
best_test_mse = 10.
#Loop through the etas and lambdas
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #Make neural network
        ffnn = NeuralNetwork(X_train, 
                             y_train, 
                             batch_size=int(n_samples/m), 
                             n_categories = 1,
                             epochs = 100, 
                             n_hidden_neurons = 20, 
                             eta = eta,
                             lmbd = lmbd,
                             input_activation = 'tanh',
                             output_activation = 'tanh',
                             cost_function = 'regression')
        ffnn.add_layer(20, activation_method = 'tanh')
        ffnn.add_layer(20, activation_method = 'tanh')
        
        #Train network
        ffnn.train()
        
        #Save predictions
        y_tilde_train = ffnn.predict(X_train)
        y_tilde_test = ffnn.predict(X_test)
        
        #Save metrics into exportable matrices
        train_mse[i][j], train_R2[i][j] = statistics.calc_statistics(y_train, y_tilde_train)
        test_mse[i][j], test_R2[i][j] = statistics.calc_statistics(y_test, y_tilde_test)
        
        if best_train_mse > train_mse[i][j]:
            best_train_mse = train_mse[i][j]
            best_y_tilde_train = y_tilde_train
        
        if best_test_mse > test_mse[i][j]:
            best_test_mse = test_mse[i][j]
            best_y_tilde_test = y_tilde_test
            
        
        #print some outputs
        print('Learning rate: ', eta)
        print('lambda: ', lmbd)
        #print('rocauc: ', test_rocauc[i][j])
        #print('accuracy: ', test_accuracy[i][j])
        print('Train. mse = ', train_mse[i][j], 'R2 = ', train_R2[i][j])
        print('Test. mse = ', test_mse[i][j], 'R2 = ', test_R2[i][j])
        print('\n')
            
            
#Visualization        
show_heatmap_mse_R2(lmbd_vals, eta_vals, train_mse, test_mse, train_R2, test_R2)

#Rescale for plotting
rescaled_dataset = FrankeDS.rescale_back(x = FrankeDS.test_x_1d, y = best_y_tilde_test)
x, y, z = rescaled_dataset[:,0], rescaled_dataset[:,1], rescaled_dataset[:,2] 

#generate Franke for plotting
Frankeplot = dataset(0)
Frankeplot.generate_franke(100, 0)
an_x, an_y = Frankeplot.x0_mesh, Frankeplot.x1_mesh
an_z = Frankeplot.y_mesh

#Plot best fit
plot_3d(x, y, z, an_x, an_y, an_z)





