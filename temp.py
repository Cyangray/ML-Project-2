from visualization import show_heatmaps
import numpy as np

show_heatmaps(lmbd_vals, 
              eta_vals, 
              train_accuracy, 
              test_accuracy, 
              train_rocauc, 
              test_rocauc, 
              train_area_ratio, 
              test_area_ratio)
