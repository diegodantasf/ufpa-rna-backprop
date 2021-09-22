from src.network_v2 import Network
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 1000
TEST_DATASET_SIZE = 1000
VALIDATION_DATASET_SIZE = 1000
N_EPOCHS = 10000
N_NEURONS = 64

def squared(x):
    return x**2

FUNCTION = squared
LOW = -10
HIGH = 10

def main():
    train_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    validation_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=VALIDATION_DATASET_SIZE, with_noise=False, seed=333)

    net = Network([1, N_NEURONS, 1])

    train_loss, test_loss = net.train(
        train_data.X, 
        train_data.y, 
        N_EPOCHS, 
        lr=0.0001, 
        test_X=test_data.X, 
        test_y=test_data.y, 
        cross_validation=True
    )

    for i in range(len(net.weights)):
        print (f'Weights at layer {i}')
        print(net.weights[i].shape)
        print(net.weights)
    
    for i in range(len(net.biases)):
        print (f'Biases at layer {i}')
        print(net.biases[i].shape)
        print(net.biases)

    plot_losses(train_loss, test_loss, filename='losses_test-loss-{}.png'.format(test_loss[-1]))
    y_pred = net.predict(validation_data.X).reshape(-1)
    plot_predictions(validation_data, y_pred, filename='predictions_test-loss-{}.png'.format(test_loss[-1]))


    
    
if __name__ == '__main__':
    main()