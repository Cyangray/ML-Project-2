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
from functions import discretize, make_onehot, inverse_onehot
from neural_network import NeuralNetwork
import seaborn as sns



#k-fold cross validation parameters
CV = False
k = 5

#random dataset, or credit card?
randomdataset = False

#Descent method
method = 'logreg'

#Polynomial features? If not, deg = 0
deg = 0

#Stochastic gradient descent parameters
m = 20           #Number of minibatches
Niterations = 1000
n_samples = 100000


if randomdataset:
    X, y = datasets.make_classification(n_samples=n_samples, n_features=20,
                                        n_informative=2, n_redundant=2)
    CDds = dataset(0)
    val = np.column_stack((X,y))
    CDds.values = val
    liste = [CDds]
    
    #polishing the dataset, and divide into data and target data
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
    
    #Make Neural Network
    X_train = CDds.x_1d
    y_train = CDds.y_1d[:,np.newaxis]
    y_train_onehot = make_onehot(y_train)
    
    #grid search
    #eta_vals = np.logspace(-7, -1, 7)
    eta_vals = np.linspace(1e-4, 1e-3, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    FFNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    train_rocauc = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_rocauc = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            ffnn = NeuralNetwork(X_train, 
                                 y_train_onehot, 
                                 batch_size=int(n_samples/m), 
                                 n_categories = 2, 
                                 epochs = 30, 
                                 n_hidden_neurons = 7, 
                                 eta = eta,
                                 lmbd = lmbd)
            ffnn.train()
            FFNN_numpy[i,j] = ffnn
            y_tilde_train = ffnn.predict_probabilities(X_train)
            
            y_tilde = ffnn.predict_probabilities(CDds.test_x_1d)
            
            _, target = CDds.rescale_back(x = CDds.test_x_1d, y = CDds.test_y_1d, split = True)
            target = [int(elem) for elem in target]
            
            train_accuracy[i][j] = statistics.calc_accuracy(y_train, y_tilde_train)
            test_accuracy[i][j] = statistics.calc_accuracy(target, y_tilde)
            train_rocauc[i][j] = roc_auc_score(y_train, y_tilde_train)
            test_rocauc[i][j] = roc_auc_score(target, y_tilde)
            
            print('Learning rate: ', eta)
            print('lambda: ', lmbd)
            print('rocauc: ', test_rocauc[i][j])
            
            
#Visualization

sns.set()

        
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_rocauc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Train ROC-AUC score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_rocauc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test ROC-AUC score")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()