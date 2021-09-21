import numpy as np

from src.helper import *

class Network(object):
    def __init__(self, layers_sizes, seed=42):
        self.rng = np.random.default_rng(seed)
        self.layers_sizes = layers_sizes
        self.biases = np.array([self.rng.normal(size=(y, 1)) for y in layers_sizes[1:]])
        self.weights = np.array([self.rng.normal(size=(y, x)) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])])

    def train(self, train_X, train_y, epochs, 
        lr=0.001, test_X = None, test_y = None, cross_validation=None):
        
        return self._gradient_descent(
            train_X, 
            train_y, 
            epochs, 
            lr=lr, 
            test_X=test_X, 
            test_y=test_y, 
            cross_validation=cross_validation
        )
    
    def evaluate(self, X, y):
        results = self._forward(X)
        ans = mean_squared_error(y, results)
        return ans
    
    def predict(self, X):
        return self._forward(X)

    def _forward(self, X):
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            X = tanh(np.dot(w, X) + b)
            X = np.dot(self.weights[-1], X) + self.biases[-1]
        return X

    def _gradient_descent(self, train_X, train_y, epochs,
        lr=0.001, test_X = None, test_y = None, cross_validation=None):
        
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        train_losses = []
        test_losses = []

        if cross_validation:
            test_X = np.array(test_X)
            test_y = np.array(test_y)
            
            train_X = np.transpose(train_X)
            test_X = np.transpose(test_X)

        for e in range(epochs):
        
            self._update_weights(train_X, train_y, lr)

            train_loss = self.evaluate(train_X, train_y)
            train_losses.append(train_loss)

            print('[TREINO] Erro na epoca {}/{}: {}'.format(e, epochs, train_loss))

            if cross_validation:
                test_loss = self.evaluate(test_X, test_y)
                test_losses.append(test_loss)
                print('[VALIDACAO-CRUZADA] Erro na epoca {}/{}: {}'.format(e, epochs, test_loss))
        
        return (np.array(train_losses), np.array(test_losses))

    def _update_weights(self, data_X, data_y, lr):
        new_b, new_w = self._backprop(data_X, data_y)
        self.weights = np.subtract(self.weights, np.multiply(lr, new_w))
        self.biases = np.subtract(self.biases, np.multiply(lr, new_b))
    
    def _backprop(self, x, y):
        local_grad_b = [np.zeros(b.shape) for b in self.biases]
        local_grad_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]

        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = tanh(z)
            activations.append(activation)
        
        # backward
        delta = mean_squared_error_prime(y, activations[-1]) * tanh_prime(zs[-1])
        local_grad_b[-1] = delta
        local_grad_w[-1] = np.dot(delta, np.transpose(activations[-2]))

        for l in range(2, len(self.layers_sizes)):
            z = zs[-l]
            tp = tanh_prime(z)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * tp
            local_grad_b[-l] = delta
            local_grad_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        
        return (local_grad_b, local_grad_w)