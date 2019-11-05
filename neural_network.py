import numpy as np
from functions import sigmoid

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=2,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.n_layers = 0
        self.layers = []

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.debug = True
        self.add_layer(n_hidden_neurons)
        


        
    def add_layer(self, n_neurons):
        #Create a new layer
        self.n_layers += 1
        if self.n_layers == 1:
            n_features = self.n_features
        else:
            n_features = self.layers[-1].n_neurons
        
        current_layer = layer(n_neurons, n_features = n_features, n_categories = self.n_categories)
        self.layers.append(current_layer)
        
        #Initialize new layer
        current_layer.create_biases_and_weights()
        
    def close_last_layer(self):
        last_layer = self.layers[-1]
        last_layer.output_weights = np.random.randn(last_layer.n_hidden_neurons, self.n_categories)
        last_layer.output_bias = np.zeros(self.n_categories) + 0.01
        
    def feed_forward(self):
        # feed-forward for training
        
        #hidden layers
        for i, current_layer in enumerate(self.layers):
            if i == 0:
                #Input layer
                current_layer.z_h = np.matmul(self.X_data, current_layer.hidden_weights) + current_layer.hidden_bias
            else:
                #hidden layer
                prev_layer = self.layers[i-1]
                current_layer.z_h = np.matmul(prev_layer.a_h, current_layer.hidden_weights) + current_layer.hidden_bias
            
            current_layer.a_h = sigmoid(current_layer.z_h)
            
            if i == len(self.layers):
                #####FIKS HER####
                
                prev_layer.z_o = current_layer.z_h
                exp_term = np.exp(prev_layer.z_o)
                prev_layer.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            
            if self.debug:
                current_layer.saved_probs = current_layer.probabilities.copy()
                self.debug = False

    def feed_forward_out(self, X):
        # feed-forward for output
        last_layer = self.layers[-1]
        z_h = np.matmul(X, last_layer.hidden_weights) + last_layer.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, last_layer.output_weights) + last_layer.output_bias
        
        exp_term = np.exp(z_o)
        
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        # Check both the math, and make a scheme of the variables, how they effect each other. 
        #This will also be useful for the report. Multilevel, and the trivial for
        #only one hidden layer
        last_layer = self.layers[-1]
        for i, current_layer in reversed(list(enumerate(self.layers))):
            
            if current_layer == last_layer:
                #last layer
                error_output = current_layer.probabilities - self.Y_data
                last_layer.output_weights_gradient = np.matmul(last_layer.a_h.T, error_output)
                last_layer.output_bias_gradient = np.sum(error_output, axis=0)
            else:
                error_output = current_layer.probabilities - self.layers[i+1].error_hidden
                
            current_layer.error_hidden = np.matmul(error_output, current_layer.output_weights.T) * current_layer.a_h * (1 - current_layer.a_h)
            
        
            current_layer.hidden_weights_gradient = np.matmul(current_layer.a_h, current_layer.error_hidden)
            #Previous line. Check that the first layer points to X_data when calling a_h
            #current_layer.hidden_weights_gradient = np.matmul(current_layer.X_data.T, current_layer.error_hidden)
            current_layer.hidden_bias_gradient = np.sum(current_layer.error_hidden, axis=0)
        
            if self.lmbd > 0.0:
                if current_layer == last_layer:
                    current_layer.output_weights_gradient += self.lmbd * current_layer.output_weights
                current_layer.hidden_weights_gradient += self.lmbd * current_layer.hidden_weights
            
            if current_layer == last_layer:
                current_layer.output_weights -= self.eta * current_layer.output_weights_gradient
                current_layer.output_bias -= self.eta * current_layer.output_bias_gradient
            current_layer.hidden_weights -= self.eta * current_layer.hidden_weights_gradient
            current_layer.hidden_bias -= self.eta * current_layer.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        self.probs = probabilities
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                #return 0
                self.backpropagation()
                
class layer():
    def __init__(self, n_neurons, n_features):
        self.n_hidden_neurons = n_neurons
        self.n_features = n_features
        
    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01
        
        
        
        
        
        
        
        
        
        
        