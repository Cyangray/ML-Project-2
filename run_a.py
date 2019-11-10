import numpy as np
from visualization import plot_features, plot_correlation_matrix
from dataset_objects import credit_card_dataset

#Importing the credit card dataset
filename = "credit_card_data_set.xls"
CDds = credit_card_dataset(filename)
liste = [CDds]
#polishing the dataset, divide into data and target data, do not drop values
CDds.CreditCardPolish(drop0=False)

#Amount of occurrences before polishing
unique, counts = np.unique(CDds.y_1d, return_counts=True)
length = np.sum(counts)
occurrences = dict(zip(unique, counts))
print(occurrences)
print('Percent of ', unique[0], ': ', occurrences[unique[0]]/length, 'before polishing')
print('Percent of ', unique[1], ': ', occurrences[unique[1]]/length, 'before polishing')

#polishing the dataset, divide into data and target data, drop values
CDds.CreditCardPolish()

#Amount of occurrences after polishing
unique, counts = np.unique(CDds.y_1d, return_counts=True)
length = np.sum(counts)
occurrences = dict(zip(unique, counts))
print(occurrences)
print('Percent of ', unique[0], ': ', occurrences[unique[0]]/length, 'after polishing')
print('Percent of ', unique[1], ': ', occurrences[unique[1]]/length, 'after polishing')
    

#Plot features
CDds.plot_setup()
#plot_features(CDds)

#Normalize dataset
CDds.normalize_dataset()

#Plot correlation matrix
plot_correlation_matrix(CDds)