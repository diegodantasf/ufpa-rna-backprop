import numpy as np

from src.helper import *

class Network(object):
    def __init__(self, layers_sizes, seed=42):
        self.rng = np.random.default_rng(seed)
        self.layers_sizes = layers_sizes
        self.biases = [self.rng.normal(size=(y, 1)) for y in layers_sizes[1:]]
        self.weights = [self.rng.normal(size=(y, x)) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]

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
        return mean_squared_error(y, results)
    
    def predict(self, X):
        return self._forward(X)

    def _forward(self, X):
        X = np.transpose(X)
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            X = tanh(np.dot(w, X) + b)
        X = np.dot(self.weights[-1], X) + self.biases[-1]
        return X

    def _gradient_descent(self, train_X, train_y, epochs,
        lr=0.001, test_X = None, test_y = None, cross_validation=None):

        train_losses = []
        test_losses = []

        for e in range(epochs):
        
            self._update_weights(train_X, train_y, lr)

            train_loss = self.evaluate(train_X, train_y)
            train_losses.append(train_loss)

            print('[TREINO] Erro na epoca {}/{}: {}'.format(e + 1, epochs, train_loss))

            if cross_validation:
                test_loss = self.evaluate(test_X, test_y)
                test_losses.append(test_loss)
                
                print('[VALIDACAO-CRUZADA] Erro na epoca {}/{}: {}'.format(e + 1, epochs, test_loss))
        
        return (np.array(train_losses), np.array(test_losses))

    def _update_weights(self, data_X, data_y, lr):
        grad_b, grad_w = self._backprop(data_X, data_y)

        for i in range(len(grad_w)):            
            grad_w[i] = np.mean(grad_w[i], axis=1).reshape(-1, 1)
            if grad_w[i].shape != self.weights[i].shape:
                grad_w[i] = np.transpose(grad_w[i])
            self.weights[i] = np.subtract(self.weights[i], np.multiply(lr, grad_w[i]))
            
        for i in range(len(grad_b)):
            grad_b[i] = np.mean(grad_b[i], axis=1).reshape(-1, 1)
            self.biases[i] = np.subtract(self.biases[i], np.multiply(lr, grad_b[i]))
                
    def _backprop(self, X, y):
        local_grad_b = [np.zeros(b.shape) for b in self.biases]
        local_grad_w = [np.zeros(w.shape) for w in self.weights]

        X = np.transpose(X)

        # feedforward
        activation = X
        activations = [X]

        zs = []
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.matmul(w, activation) + b
            zs.append(z)
            if i != len(self.biases) - 1:
                activation = tanh(z)
            else:
                activation = z
            activations.append(activation)
        
        # backward
        delta = mean_squared_error_prime(y, activations[-1])
        local_grad_b[-1] = delta
        local_grad_w[-1] = np.multiply(delta, activations[-2])


        for l in range(2, len(self.layers_sizes)):
            z = zs[-l]
            tp = tanh_prime(z)
            delta = np.transpose(self.weights[-l+1]) * delta * tp
            local_grad_b[-l] = delta
            local_grad_w[-l] = np.multiply(delta, activations[-l-1])

        return (local_grad_b, local_grad_w)