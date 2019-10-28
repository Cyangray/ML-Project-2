import numpy as np
from sklearn.metrics import r2_score as r2


def calc_MSE(z, z_tilde):
    mse = 0
    n=len(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
    return mse/n


def calc_R2_score(z, z_tilde):
    mse = 0
    ms_avg = 0
    n=len(z)
    mean_z = np.mean(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
        ms_avg += (z[i] - mean_z)**2
    return 1. - mse/ms_avg

def calc_R2_score_sklearn(z, z_tilde):
    return r2(z, z_tilde)


def calc_statistics(z, z_tilde):
    mse = calc_MSE(z, z_tilde)
    calc_r2 = calc_R2_score(z, z_tilde)
    return mse, calc_r2

def calc_bias_variance(z, z_tilde):
    """ Calculate the bias and the variance of a given model"""
    n = len(z)
    Eztilde = np.mean(z_tilde)
    bias = 1/n * np.sum((z - Eztilde)**2)
    variance = 1/n * np.sum((z_tilde - Eztilde)**2)
    return bias, variance

def print_mse(mse):
    print("Average mse: ", np.average(mse))
    print("Best mse: ", np.min(mse[np.argmin(np.abs(np.array(mse)))]))

def print_R2(R2):
    print("Average R2: ", np.average(R2))
    print("Best R2: ", R2[np.argmax(np.array(R2))])


