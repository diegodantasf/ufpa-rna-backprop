import numpy as np

def mean_squared_error(y, y_hat):
    return np.mean((np.subtract(y, y_hat))**2)

def mean_squared_error_prime(y, y_hat):
    return np.multiply(-2, np.subtract(y, y_hat))
    
def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2