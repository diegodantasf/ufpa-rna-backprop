from src.net import RNA
from src.network import Network
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 100
TEST_DATASET_SIZE = 100
VALIDATION_DATASET_SIZE = 100
N_EPOCHS = 4000
N_NEURONS = 16

def f(x):
    return x**2

FUNCTION = np.sin
LOW = -5 * np.pi
HIGH = 10 * np.pi

def main():
    train_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    validation_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=VALIDATION_DATASET_SIZE, with_noise=False, seed=333)

    net = Network([1, N_NEURONS, 1])

    # print('weights: ', net.weights)
    # print('biases: ', net.biases)
    
    train_loss, test_loss = net.train(train_data.X, train_data.y, N_EPOCHS, 5, lr=0.0009, test_X=test_data.X, test_y=test_data.y, cross_validation=True)
    # print('weights: ', net.weights)
    # print('biases: ', net.biases)

    y_pred = net.predict(np.transpose(validation_data.X))
    y_pred = np.squeeze(y_pred)

    validation_data.X = np.squeeze(validation_data.X)

    plot_losses(train_loss, test_loss, filename='losses_test-loss-{}.png'.format(test_loss[-1]))
    plot_predictions(validation_data, y_pred, filename='predictions_test-loss-{}.png'.format(test_loss[-1]))
    
    
if __name__ == '__main__':
    main()