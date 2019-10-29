import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from visualization import plot_features
from sklearn import preprocessing
import sys
from functions import franke_function

class dataset():
    
    def __init__(self, filename, header = 0, skiprows = 0, index_col = 0):
        nanDict = {}
        self.normalized = False
        self.df = pd.read_excel(filename, header = header, skiprows = skiprows, index_col = index_col, na_values=nanDict)
        
    '''Divide the DataFrame into data and target.'''
    def polish_and_divide(self, targetcol = -1, headerrows = 0, headercols = 0):
        self.values = np.copy(self.df.to_numpy())
        if headerrows == 0:
            self.df.feature_names = np.copy(self.df.columns)
        else:
            self.df.feature_names = self.iloc[headerrows-1]
        self.feature_names = self.df.feature_names
        self.x_1d = self.values[headercols: , headerrows : targetcol]
        self.y_1d = self.values[headercols: , targetcol]
        
    def normalize_dataset(self):
        ''' Uses the scikit-learn preprocessing tool for scaling the datasets. '''
        self.normalized = True
        self.x_1d_unscaled = self.x_1d.copy()
        self.y_1d_unscaled = self.y_1d.copy()
        dataset_matrix = self.values
        self.scaler = preprocessing.StandardScaler().fit(dataset_matrix)
        transformed_matrix = self.scaler.transform(dataset_matrix)
        self.x_1d = transformed_matrix[:,:-1]
        self.y_1d = transformed_matrix[:,-1]
        
class credit_card_dataset(dataset):
    
    def __init__(self, filename):
        super().__init__(filename, header = 1, skiprows = 0, index_col = 0)
    
    def CreditCardPolish(self):
        self.df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
        super().polish_and_divide(headerrows = 0, headercols = 0)
        
    def plot_setup(self):
        self.contbins = self.df['AGE'].max() - self.df['AGE'].min()
        self.continuous_features_idxs = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
        self.cont_rows = 7
        self.cont_cols = 2
        self.discrete_features_idxs = [1,2,3,5,6,7,8,9,10]
        self.disc_rows = 3
        self.disc_cols = 3
        self.output_labels = ["Non-default", "Default"]