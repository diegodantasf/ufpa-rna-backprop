from src.network_v2 import Network
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 1000
TEST_DATASET_SIZE = 1000
VALIDATION_DATASET_SIZE = 1000
N_EPOCHS = 100
N_NEURONS = 1

FUNCTION = np.sin
LOW = -5 * np.pi
HIGH = 10 * np.pi

def main():
    train_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    validation_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=VALIDATION_DATASET_SIZE, with_noise=False, seed=333)

    net = Network([1, N_NEURONS, 1])
    
    train_loss, test_loss = net.train(train_data.X, train_data.y, N_EPOCHS, lr=0.00005, test_X=test_data.X, test_y=test_data.y, cross_validation=True)

    y_pred = net.predict(np.transpose(validation_data.X))
    y_pred = np.squeeze(y_pred)
    
    validation_data.X = np.squeeze(validation_data.X)

    print ('Network weights')
    print (net.weights.shape)
    print (net.weights)
    print ('Network biases')
    print (net.biases.shape)
    print (net.biases)

    plot_losses(train_loss, test_loss, filename='losses_test-loss-{}.png'.format(test_loss[-1]))
    plot_predictions(validation_data, y_pred, filename='predictions_test-loss-{}.png'.format(test_loss[-1]))


    
    
if __name__ == '__main__':
    main()