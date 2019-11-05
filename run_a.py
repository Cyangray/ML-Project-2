import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features, plot_correlation_matrix
from dataset_objects import dataset, credit_card_dataset

#Importing the credit card dataset
filename = "credit_card_data_set.xls"
CDds = credit_card_dataset(filename)
liste = [CDds]

#polishing the dataset, and divide into data and target data
CDds.CreditCardPolish()

unique, counts = np.unique(CDds.y_1d, return_counts=True)
length = np.sum(counts)
occurrences = dict(zip(unique, counts))
print(occurrences)
print('Percent of ', unique[0], ': ', occurrences[unique[0]]/length)
print('Percent of ', unique[1], ': ', occurrences[unique[1]]/length)
    

#Plot features
#CDds.plot_setup()
#plot_features(CDds)

#Normalize dataset
#CDds.normalize_dataset()

#Plot correlation matrix
#plot_correlation_matrix(CDds)