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
            n_features = self.layers[-1].n_hidden_neurons
        
        current_layer = layer(n_neurons, n_features)
        self.layers.append(current_layer)
        
        #Initialize new layer
        current_layer.create_biases_and_weights()
        
    def close_last_layer(self):
        last_layer = self.layers[-1]
        last_layer.output_weights = np.random.randn(last_layer.n_hidden_neurons, self.n_categories)
        last_layer.output_bias = np.zeros(self.n_categories) + 0.01
        
    def feed_forward(self, X = 0, output = False):
        # feed-forward for training
        
        #Take X as input. If none is given, take the self.X_data
        if isinstance(X, int):
            X_start = self.X_data
        else:
            X_start = X
        
        #Loop through the layers and update the z_h and a_h
        for i, current_layer in enumerate(self.layers):
            if i == 0: #Input layer: take dataset as input
                previous_a_h = X_start
            else: #hidden layer: take a_h from the previous layer as input
                previous_a_h = self.layers[i-1].a_h
                
            current_layer.z_h = np.matmul(previous_a_h, current_layer.hidden_weights) + current_layer.hidden_bias
            current_layer.a_h = sigmoid(current_layer.z_h)
        
        ### Calculate the probabilities for the output
        last_layer = self.layers[-1]
        last_layer.z_o = np.matmul(last_layer.a_h, last_layer.output_weights) + last_layer.output_bias
        exp_term = np.exp(last_layer.z_o)
        last_layer.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        
        if output: #Print last layer's probabilities
            return last_layer.probabilities

    def feed_forward_out(self, X):
        # feed-forward for output
        probabilities = self.feed_forward(X = X, output = True)
        return probabilities

    def backpropagation(self):
        #Backpropagating algorithm
        
        # Calculate output error and the weights and bias gradients
        last_layer = self.layers[-1]
        error_output = last_layer.probabilities - self.Y_data
        last_layer.output_weights_gradient = np.matmul(last_layer.a_h.T, error_output)
        last_layer.output_bias_gradient = np.sum(error_output, axis=0)
        
        if self.lmbd > 0.0:
            last_layer.output_weights_gradient += self.lmbd * last_layer.output_weights
        
        #Update the output weights and biases
        last_layer.output_weights -= self.eta * last_layer.output_weights_gradient
        last_layer.output_bias -= self.eta * last_layer.output_bias_gradient
        
        #Loop through the layers backwards in order to update the weights
        for i, current_layer in reversed(list(enumerate(self.layers))):
            #Check if we are evaluating the last or the first layer, as some variables will
            #point to different things, similarly as with the feed_forward algorithm
            if current_layer == last_layer:
                forward_error = error_output
                forward_weights = last_layer.output_weights
            else:
                forward_error = self.layers[i+1].error_hidden
                forward_weights = self.layers[i+1].hidden_weights
                
            if current_layer == self.layers[0]:
                previous_a_h = self.X_data
            else:
                previous_a_h = self.layers[i-1].a_h
            
            #calculate the error in the hidden weights, and update the gradients
            current_layer.error_hidden = np.matmul(forward_error, forward_weights.T) * current_layer.a_h * (1 - current_layer.a_h)
            current_layer.hidden_weights_gradient = np.matmul(previous_a_h.T, current_layer.error_hidden)
            current_layer.hidden_bias_gradient = np.sum(current_layer.error_hidden, axis=0)
        
            if self.lmbd > 0.0:
                current_layer.hidden_weights_gradient += self.lmbd * current_layer.hidden_weights
            
            #Update the weights and the biases
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
        self.close_last_layer()
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
        
        
        
        
        
        
        
        
        
        
        