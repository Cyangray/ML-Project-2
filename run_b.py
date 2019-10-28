import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task b) of the project.
Using k-fold cross validation for running data,
and evaluating MSE and R^2 for this sampling method."""

deg = 5                     # degree of the polynomial fit
k = 5                       # k batches for k-fold.
method = "least squares"    # "least squares", "ridge" or "lasso"

# Load previously generated dataset
dataset = data_generate()
dataset.load_data()

# Normalize the dataset and divide in samples
dataset.normalize_dataset()
dataset.sort_in_k_batches(k)


# Run k-fold algorithm and fit models.
sample = sampling(dataset)
sample.kfold_cross_validation(k, method, deg=deg)

# Calculate statistics
print("Batches: k = ", k)
statistics.print_mse(sample.mse)
statistics.print_R2(sample.R2)

# Plotting the best fit with the lowest mse.
dataset.reload_data()
fitted = fit(dataset)
fitted.create_design_matrix(deg = deg)
z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
z_model = rescaled_dataset[2]

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(n, noise=0)

# Plot
plot_3d(rescaled_dataset[0], rescaled_dataset[1], z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")