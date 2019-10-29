from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
import sys

def plot_features(dataset):
    
    targ0 = dataset.x_1d[dataset.y_1d == 0]
    targ1 = dataset.x_1d[dataset.y_1d == 1]
    cont_rows = dataset.cont_rows
    cont_cols = dataset.cont_cols
    disc_rows = dataset.disc_rows
    disc_cols = dataset.disc_cols
    contbins = dataset.contbins #79-21
    output_labels = dataset.output_labels
    
    #Plot continuous features
    fig, axes = plt.subplots(cont_rows, cont_cols, figsize=(10,20))
    ax = axes.ravel()
    
    continuous_features_idxs = dataset.continuous_features_idxs
    
    for graph, i in enumerate(continuous_features_idxs):
        _, bins = np.histogram(dataset.x_1d[:,i], bins =contbins)
        ax[graph].hist([targ0[:,i], targ1[:,i]], bins = bins, stacked = True)
        ax[graph].set_title(dataset.feature_names[i])
        ax[graph].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(output_labels, loc ="best")
    fig.tight_layout()
    plt.show()
    
    #Plot discrete features
    discrete_features_idxs = dataset.discrete_features_idxs
    '''labels = [
            ('Male', 'Female'),
            ('Unknown', 'Grad. school', 'University', 'High school', 'Others', 'Unknown', 'Unknown'),
            ('Unknown', 'Married', 'Single', 'Others'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months'),
            ('Unknown_1', 'Pay duly', 'Unknown_2', 'Delay 1 month', 'Delay 2 months', 'Delay 3 months', 'Delay 4 months', 'Delay 5 months', 'Delay 6 months', 'Delay 7 months', 'Delay 8 months', 'Delay 9+ months')
            ]'''
    
    fig, axes = plt.subplots(disc_rows, disc_cols, figsize=(10,20))
    ax = axes.ravel()
    
    width = 0.35
    for graph, i in enumerate(discrete_features_idxs): #Loop through the features
        uniques, _ = np.unique(dataset.x_1d[:,i], return_counts=True)
        uniquestarg1, countstarg1_1 = np.unique(targ1[:,i], return_counts=True)
        uniquestarg0, countstarg0_1 = np.unique(targ0[:,i], return_counts=True)
        countstarg1 = np.zeros(np.shape(uniques))
        countstarg0 = np.zeros(np.shape(uniques))
        for j, un in enumerate(uniques): #For each feature, sort which is default and which is not
            if un in uniquestarg1:
                countstarg1[j] = countstarg1_1[np.where(uniquestarg1 == un)]
            if un in uniquestarg0:
                countstarg0[j] = countstarg0_1[np.where(uniquestarg0 == un)]

        ax[graph].bar(uniques, countstarg0, width)
        ax[graph].bar(uniques, countstarg1, width, bottom = countstarg0)
        ax[graph].set_title(dataset.feature_names[i])
        ax[graph].set_yticks(())
        ax[graph].set_xticks(uniques)
    ax[0].set_ylabel("Frequency")
    ax[0].legend(output_labels, loc ="best")
    #fig.tight_layout()
    plt.show()


def plot_correlation_matrix(normalized_dataset):
    fig = plt.subplots()
    correlation_matrix = normalized_dataset.df.corr().round(1)
    # use the heatmap function from seaborn to plot the correlation matrix
    # annot = True to print the values inside the square
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.show()


def plot_3d(x, y, z, an_x, an_y, an_z, plot_type):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("The franke function model and analytical solution.", fontsize=22)
    
    # Surface of analytical solution.
    surf = ax.plot_surface(an_x, an_y, an_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf_2 = ax.scatter(x, y, z)

    # Customize the z axis.
    ax.set_zlim(-0.30, 2.40)#(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_title("The franke function model and analytical function", fontsize = 20)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
        
        
def plot_3d_terrain(x, y, z, x_map, y_map, z_map):
    """ Plots 3d terrain with trisurf"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf1 = ax.plot_trisurf(x_map, y_map, z_map, cmap=cm.coolwarm, alpha=0.2)
    surf2 = ax.scatter(x, y, z)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_bias_var_tradeoff(deg, mse):
    """ Plots bias-variance tradeoff for different polynoial degrees of models. """
    plt.title("Bias-variance tradeoff for different complexity of models")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Prediction error")
    plt.plot(deg, mse)
    plt.grid('on')
    plt.show()

def plot_mse_vs_complexity(deg, mse_test, mse_train):
    """ Plots mse vs. polynomial degree of matrix. """
    fig, ax = plt.subplots()
    ax.set_title("Bias-variance tradeoff for different complexity of models")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Prediction error")
    ax.plot(deg, mse_test, 'r-', label = 'Test sample')
    ax.plot(deg, mse_train, 'b-', label = 'Training sample')
    plt.grid('on')
    plt.legend()
    plt.show()

def plot_bias_variance_vs_complexity(deg, bias, variance):
    """ Plots bias-variance vs. polynomial degree of matrix. """
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax1.set_title("Bias-variance tradeoff for different complexity of models")
    #ax1 = plt.subplot(211)
    ax1.set_ylabel("Bias values")
    ax1.plot(deg, bias, 'r-', label = 'Bias')
    ax1.legend()
    ax1.grid('on')
    
    #ax2 = plt.subplot(212, sharex = ax1)
    ax2.set_xlabel("Polynomial degree")
    ax2.set_ylabel("Variance values")
    ax2.plot(deg, variance, 'b-', label = 'Variance')
    ax2.grid('on')
    ax2.legend()
    plt.show()

def plot_beta(lambdas, beta):
    """ Plots the betas in function of the hyperparameter lambda"""
    fig, ax = plt.subplots()
    ax.set_title("values of beta as function of lambda")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("beta")
    labels = ["beta_" + str(i) for i in range(len(beta[0,:]))]
    
    for i in range(len(beta[0,:])):
        ax.plot(lambdas, beta[:,i], 'r-', label = labels[i])
    plt.xscale('log')
    plt.grid('on')
    plt.show()
    
def plot_bias_variance_vs_lambdas(lambdas, mse_test, mse_train):
    """ Plots mse vs. values of lambda. """
    fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax1.set_title("MSE for different lambdas")
    ax1.set_xscale('log')
    ax1.set_ylabel("Prediction error")
    ax1.plot(lambdas, mse_test, 'r-', label = 'Bias')
    ax1.grid('on')
    ax1.legend()
    
    
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("Prediction error")
    ax2.plot(lambdas, mse_train, 'b-', label = 'Variance')
    ax2.set_xscale('log')
    ax2.grid('on')
    ax2.legend()
    plt.show()
    
    
def plot_mse_vs_lambda(lambdas, mse_test, mse_train):
    """ Plots mse vs. lambdas. """
    fig, ax = plt.subplots()
    ax.set_title("Bias-variance tradeoff for different lambdas")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Prediction error")
    ax.plot(lambdas, [mse_test[i] - mse_train[i] for i in range(len(mse_test))], 'r-', label = 'Test sample')
    #ax.plot(lambdas, mse_train, 'b-', label = 'Training sample')
    ax.set_xscale('log')
    plt.grid('on')
    plt.legend()
    plt.show()


def plot_terrains(ind_var, ind_var_text, method, CV_text, x_matrices, x_labels, mses, R2s, lambdas, biases, variances):
    """ Function for plotting various useful plots from the real terrain data. """
    # Plot terrains
    fig, axs = plt.subplots(nrows = 1, ncols = len(ind_var), sharey = True)
    xlabels = [ind_var_text + " = " + str(i) for i in ind_var]
    axs[2].set_title("Model of map for " + method + ", " + CV_text + " cross validation")
    for i, ax in enumerate(axs):
        ax.imshow(z_matrices[i], cmap = cm.coolwarm)
        ax.set_xlabel(xlabels[i])
    plt.show()

    # Plot errors
    fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax1.set_title("MSE and R2 score of map for " + method + ", " + CV_text + " cross validation")
    ax1.plot(ind_var,mses,'r-',label = "MSE")
    if ind_var == lambdas:
        ax1.set_xscale('log')
    ax1.grid('on')
    ax1.legend()

    ax2.set_xlabel(ind_var_text)
    ax2.plot(ind_var,R2s,'b-',label = "R2 score")
    if ind_var == lambdas:
        ax2.set_xscale('log')
    ax2.grid('on')
    ax2.legend()
    plt.show()

    #Plot bias and variance
    fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    ax1.set_title("Bias and variance of map for " + method + ", " + CV_text + " cross validation")
    ax1.plot(ind_var,biases,'r-',label = "Bias")
    if ind_var == lambdas:
        ax1.set_xscale('log')
    ax1.grid('on')
    ax1.legend()

    ax2.set_xlabel(ind_var_text)
    ax2.plot(ind_var,variances,'b-',label = "Variance")
    if ind_var == lambdas:
        ax2.set_xscale('log')
    ax2.grid('on')
    ax2.legend()
    plt.show()
        
        