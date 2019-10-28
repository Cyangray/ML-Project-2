from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys


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
        
        