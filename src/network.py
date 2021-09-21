import numpy as np
import random

class Network(object):
  def __init__(self, layers_sizes, seed=42):
    self.num_layers = len(layers_sizes)
    self.layers_sizes = layers_sizes
    self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]
    self.biases = np.array(self.biases)
    self.weights = np.array(self.weights)
  

  def train(self, train_X, train_y, epochs, mini_batch_size, lr=0.001, test_X = None, test_y = None, cross_validation=None):
    return self.SDG(train_X, train_y, epochs, mini_batch_size, lr=lr, test_X=test_X, test_y=test_y, cross_validation=cross_validation)

  def forward(self, X):
    for b, w in zip(self.biases[:-1], self.weights[:-1]):
      X = tanh(np.dot(w, X) + b)
    X = np.dot(self.weights[-1], X) + self.biases[-1]
  
    return X
  
  def evaluate(self, X, y):
    results = self.forward(X)
    ans = MSE(y, results)
    return ans
  
  def predict(self, X):
    return self.forward(X)

  def shuffle_data(self, X, y):
    X, y = np.array(X), np.array(y)
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], y[idx]

  def SDG(self, train_X, train_y, epochs, mini_batch_size, lr=0.001, test_X = None, test_y = None, cross_validation=None):
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
      
      self.update_weights(train_X, train_y, lr)

      train_loss = self.evaluate(train_X, train_y)
      train_losses.append(train_loss)

      print('[TREINO] Erro na epoca {}/{}: {}'.format(e, epochs, train_loss))

      if cross_validation:
        test_loss = self.evaluate(test_X, test_y)
        test_losses.append(test_loss)
        print('[VALIDACAO-CRUZADA] Erro na epoca {}/{}: {}'.format(e, epochs, test_loss))
      else:
        print('Epoch {} finalizada'.format(e))

      print('')
    
    return (np.array(train_losses), np.array(test_losses))

  def update_weights(self, data_X, data_y, lr):

    delta_new_b, delta_new_w = self.backprop(data_X, data_y)
    new_b = delta_new_b
    new_w = delta_new_w

    self.weights = np.subtract(self.weights, np.multiply(lr, new_w))
    self.biases = np.subtract(self.biases, np.multiply(lr, new_b))
  
  def backprop(self, x, y):
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
    delta = MSE_derivative(y, activations[-1]) * tanh_prime(zs[-1])
    local_grad_b[-1] = delta
    local_grad_w[-1] = np.dot(delta, np.transpose(activations[-2]))

    for l in range(2, self.num_layers):
      z = zs[-l]
      tp = tanh_prime(z)
      delta = np.dot(np.transpose(self.weights[-l+1]), delta) * tp
      local_grad_b[-l] = delta
      local_grad_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
    
    return (local_grad_b, local_grad_w)

def MSE(y, y_hat):
    return np.mean((np.subtract(y, y_hat))**2)

def MSE_derivative(y, y_hat):
    return np.multiply(-2, np.subtract(y, y_hat))
    
def tanh(z):
  return np.tanh(z)

def tanh_prime(z):
  return 1 - np.tanh(z)**2