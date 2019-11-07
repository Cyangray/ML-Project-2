import numpy as np
import sys
from sklearn import preprocessing
from sklearn.linear_model import Lasso, SGDRegressor
from functions import sigmoid

import statistical_functions as statistics

class fit():
    def __init__(self, inst): 
        self.inst = inst
        
    def create_design_matrix(self, x = 0, N = 0, deg = 0):
        """ Function for creating a design X-matrix.
        if deg > 0, a polynomial matrix of degree 'deg' for two variables will be 
        created, with rows [1, x, y, x^2, xy, xy^2 , etc.]
        if deg == 0, a simple design matrix will be created. Useful for
        big datasets.
        Input for x is a dataset in the form of x_1d.
        Keyword argument deg is the degree of the polynomial you want to fit. """
        
        if deg == 0:
            X = self.create_simple_design_matrix(x = x)
            return X
        else:
            X = self.create_polynomial_design_matrix(x = x, N = N, deg = deg)
            return X

    def create_simple_design_matrix(self, x = 0):
        ''' Create simple design matrix from a matrix of data. If x = 0, it will
        use the x_1d attribute of the imported dataset'''
        
        if isinstance(x, int):
            self.X = self.inst.x_1d
        else:
            self.X = x
        return self.X 
        
    def create_polynomial_design_matrix(self, x=0, N=0, deg=17):
        ''' Create a polynomial design matrix from a matrix of data. If x = 0, it will
        use the x_1d attribute of the imported dataset'''
        
        if isinstance(x, int):
            x = self.inst.x_1d
            N = self.inst.N

        self.x = x
        N = x.shape[0]
        
        self.l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, self.l))
        
        #fit polynomial
        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x[:,0]**(i - k) + x[:,1]**k
                    
        #Design matrix
        self.X = X
        return X
    
    def fit_design_matrix_logistic_regression(self, descent_method = 'SGD-skl', eta = 0.000005, Niteration = 200, m = 5, verbose = False):
        '''solve the model using logistic regression. 
        Method 'SGD-skl' for SGD scikit-learn,
        method 'GD' for plain gradient descent'''
        
        n, p = np.shape(self.X)
        if descent_method == 'skl-SGD':
            if self.inst.normalized:
                fit_intercept = False
            else:
                fit_intercept = True
            sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1, fit_intercept = fit_intercept)
            sgdreg.fit(self.X, self.inst.y_1d.ravel())
            self.betas = sgdreg.coef_
            self.y_tilde = self.X@self.betas
            if verbose:
                # Cost function
                m = self.X.shape[0]
                #cost = -(1 / m) * np.sum(y * np.log(prob) + (1 - y) * np.log(compl_prob))
                cost = - (1 / m) * np.sum(self.inst.y_1d.ravel() * self.y_tilde + np.log(sigmoid(-self.y_tilde)))
                print('cost is', cost)
                
            return self.y_tilde, sgdreg.coef_
        
        elif descent_method == 'GD':
            beta = np.ones((p, 1))
            X = self.X
            y = self.inst.y_1d[:, np.newaxis]
            for iter in range(Niteration):
                y_tilde_iter = X @ beta
                prob = sigmoid(y_tilde_iter)
                compl_prob = sigmoid(-y_tilde_iter)
                gradients =  - np.transpose(X) @ (y - prob)
                
                beta -= eta*gradients
                
                if verbose:
                    # Cost function
                    m = X.shape[0]
                    #cost = -(1 / m) * np.sum(y * np.log(prob) + (1 - y) * np.log(compl_prob))
                    cost = - (1 / m) * np.sum(y * y_tilde_iter + np.log(compl_prob))
                    print('cost is', cost)
            self.betas = beta
            self.y_tilde = self.X @ beta
            return self.y_tilde, self.betas
        
        elif descent_method == 'SGD':
            #implement own stochastic gradient descent
            self.inst.sort_in_k_batches(m, random=True, minibatches = True)
            
            epochs = int(Niteration / m)
            beta = np.ones((p, 1))
            for epoch in range(1, epochs + 1):
                for i in range(m):
                    
                    #Pick random minibatch
                    minibatch_k = np.random.randint(m)
                    minibatch_data_idxs = self.inst.m_idxs[minibatch_k]
                    minibatch_data = self.inst.values[minibatch_data_idxs]
                    minibatch_x_1d = minibatch_data[:,:-1]
                    minibatch_y_1d = minibatch_data[:,-1]
                    
                    
                    X = self.create_design_matrix(x = minibatch_x_1d, deg = deg)
                    y = minibatch_y_1d[:,np.newaxis]
                    y_tilde_iter = X @ beta
                    prob = sigmoid(y_tilde_iter)
                    compl_prob = sigmoid(-y_tilde_iter)
                    gradients =  - np.transpose(X) @ (y - prob)
                    
                    beta -= eta*gradients
                    
                    if verbose:
                        # Cost function
                        m = X.shape[0]
                        #cost = -(1 / m) * np.sum(y * np.log(prob) + (1 - y) * np.log(compl_prob))
                        cost = - (1 / m) * np.sum(y * y_tilde_iter + np.log(compl_prob))
                        print('cost is', cost)
            self.betas = beta
            self.y_tilde = self.X @ beta
            return self.y_tilde, self.betas
        
    
    def fit_design_matrix_numpy(self):
        """Method that uses the design matrix to find the coefficients beta, and
        thus the prediction y_tilde"""
        X = self.X
        y = self.y
        
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_ridge(self, lambd):
        """Method that uses the design matrix to find the coefficients beta with 
        the ridge method, and thus the prediction y_tilde"""
        X = self.X
        y = self.y

        beta = np.linalg.pinv(X.T.dot(X) + lambd*np.identity(self.l)).dot(X.T).dot(y)
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_lasso(self, lambd):
        """The lasso regression algorithm implemented from scikit learn."""
        lasso = Lasso(alpha = lambd, max_iter = 10e5, tol = 0.01, normalize= (not self.inst.normalized), fit_intercept=(not self.inst.normalized))
        lasso.fit(self.X,self.y)
        beta = lasso.coef_
        y_tilde = self.X@beta
        return y_tilde, beta

    def test_design_matrix(self, beta, X = 0):
        """Testing a design matrix with beta"""
        if isinstance(X, int):
            X = self.X
        y_tilde = X @ beta
        return y_tilde
        
