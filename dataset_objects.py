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
        self.resort = 0
        self.header = header
        self.skiprows = skiprows
        if filename != 0:
            self.df = pd.read_excel(filename, header = header, skiprows = skiprows, index_col = index_col, na_values=nanDict)
            self.pandas_df = True
        else:
            self.pandas_df = False
        
    '''Divide the DataFrame into data and target.'''
    def polish_and_divide(self, targetcol = -1, headercols = 0):
        headerrows = self.header
        if self.pandas_df:
            self.values = np.copy(self.df.to_numpy())
            self.feature_names = np.copy(self.df.columns)
            
        self.x_1d = self.values[headercols: , headerrows : targetcol]
        self.y_1d = self.values[headercols: , targetcol]
        self.N = self.x_1d.shape[0]
        
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
        
    def rescale_back(self, x=0, y=0, split = False):
        """ After processing, the data must be scaled back to normal by scalers 
        inverse_transform for mainly plotting and validating purposes."""
        #self.normalized = False
        if isinstance(x, int):
            x = self.x_1d
        if isinstance(y, int):
            y = self.y_1d
        dataset_matrix = np.column_stack((x, y))
        rescaled_matrix = self.scaler.inverse_transform(dataset_matrix)
        if split:
            x_out = rescaled_matrix[:,:-1]
            y_out = rescaled_matrix[:,-1]
            return x_out, y_out
        else:
            return rescaled_matrix
    
    def sort_train_test(self, ratio=0.2, random=True):
        '''sorts the dataset into a training and a test set. Ratio is a number
        between 0 and 1, giving the ratio of the test set'''
        N_test = int(self.N*ratio)
        self.training_indices = []
        self.test_indices = []
        
        if random:
            '''Loop all indexes, Generate a random number, see if it lies below
            a treshold given by the ratio. if so, this will end up in the training set'''
            idx = 0
            for idx in range(self.N):
                random_number = np.random.rand()
                if random_number < ratio:
                    self.test_indices.append(idx)
                else:
                    self.training_indices.append(idx)
        else:
            '''shuffles randomly and splits into train and test.'''
            split = np.arange(self.N)
            np.random.shuffle(split)
            self.training_indices = split[N_test:]
            self.test_indices = split[:N_test]
            
        self.fill_array_test_training()
            
        
    def sort_in_k_batches(self, k, random=True, minibatches = False):
        """ Sorts the data into k batches, i.e. prepares the data for k-fold cross
        validation. Recommended numbers are k = 3, 4 or 5. "random" sorts the
        dataset randomly. if random==False, it sorts them statistically"""
            
        if minibatches:
            self.m = k
            self.m_idxs = [[] for i in range(k)]
        else:
            self.k = k
            self.k_idxs = [[] for i in range(k)]
            
        idx = 0
        N = self.N
        
        
        limits = [i/k for i in range(k+1)]
        
        if random:
            '''Loop all indexes, Generate a random number, see where it lies in k 
            evenly spaced intervals, use that to determine in which set to put
            each index'''
            while idx < N:
                random_number = np.random.rand()
                for i in range(k):
                    if limits[i] <= random_number < limits[i+1]:
                        if minibatches:
                            self.m_idxs[i].append(idx)
                        else:
                            self.k_idxs[i].append(idx)
                        break
                idx += 1
            
        else: 
            '''Statistical sorting lists int values, shuffles randomly and splits into k pieces.'''
            split = np.arange(N)
            np.random.shuffle(split)
            #exp_limits = [elem * N for elem in limits] 
            limits = [int(elem*N) for elem in limits]
            for i in range(k):
                if minibatches:
                    self.m_idxs[i].append( split[limits[i] : limits[i+1]] )
                else:
                    self.k_idxs[i].append( split[limits[i] : limits[i+1]] )
                
    def sort_training_test_kfold(self, i):
        """After sorting the dataset into k batches, pick one of them and this one 
        will play the part of the test set, while the rest will end up being 
        the training set. the input i should be an integer between 0 and k-1, and it
        picks the test set. """
        self.test_indices = self.k_idxs[i]
        self.training_indices = []
        for idx in range(self.k):
            if idx != i:
                self.training_indices += self.k_idxs[idx]


    def fill_array_test_training(self):
        """ Fill the arrays, eg. test_x_1d and x_1d for x, y and z with
        the actual training data according to how the indicies was sorted in 
        sort_training_test_kfold."""
        testing = self.test_indices ; training = self.training_indices

        self.reload_data()

        self.test_x_1d = self.x_1d[testing, :]
        self.test_y_1d = self.y_1d[testing]
        
        self.x_1d = self.x_1d[training, :]
        self.y_1d = self.y_1d[training]
        
        # Redefine lengths for training and testing.
        self.N = len(training)
        self.N_testing = len(testing)
        
        
    def reload_data(self):
        """ Neat system for automatically make a backup of data sets if you resort. """
        if self.resort < 1:
            np.savez("backup_data", N=self.N, x=self.x_1d, y=self.y_1d)
        else: # self.resort >= 1:
            data = np.load("backup_data.npz")
            self.N = data["N"]
            self.x_1d = data["x"]
            self.y_1d = data["y"]
        self.resort = 10
        
        
        
class credit_card_dataset(dataset):
    '''Child class of dataset, giving more methods, specific for the credit card
    dataset, with some hard-coded, dataset-specific values to give a neater experience in interface'''
    def __init__(self, filename):
        super().__init__(filename, header = 0, skiprows = 1, index_col = 0)
    
    def CreditCardPolish(self):
        self.df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
        super().polish_and_divide()
        
    def plot_setup(self):
        self.contbins = self.df['AGE'].max() - self.df['AGE'].min()
        self.continuous_features_idxs = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
        self.cont_rows = 7
        self.cont_cols = 2
        self.discrete_features_idxs = [1,2,3,5,6,7,8,9,10]
        self.disc_rows = 3
        self.disc_cols = 3
        self.output_labels = ["Non-default", "Default"]