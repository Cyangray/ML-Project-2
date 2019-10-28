import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task a) of the project.
Using generated dataset from run_generate_dataset.py, with
a dataset from Franke function with background noise
for standard least square regression w/polynomials up to the
fifth order. Also adding MSE and R^2 score."""

# Load data from previously saved file
deg = 5
n = 100
#dataset = data_generate()
#dataset.load_data()

# Or you can generate directly.
dataset = data_generate()
liste = [dataset]
dataset.generate_franke(n=n, noise=0.2)

# Normalize the dataset
dataset.normalize_dataset()

# Fit design matrix
fitted_model = fit(dataset)

# Ordinary least square fitting
fitted_model.create_design_matrix(deg)
y_model_norm, beta = fitted_model.fit_design_matrix_numpy()

# Statistical evaluation
mse, calc_r2 = statistics.calc_statistics(dataset.y_1d, y_model_norm)
print("Mean square error: ", mse, "\n", "R2 score: ", calc_r2)

# Scale back the dataset
rescaled_dataset = dataset.rescale_back(y = y_model_norm)
y_model = rescaled_dataset[:,2]

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(n, noise=0)

# Plot solutions and analytical for comparison
plot_3d(dataset.x_unscaled[:,0], dataset.x_unscaled[:,1], y_model, analytical.x0_mesh, analytical.x1_mesh, analytical.y_mesh, ["surface", "scatter"])