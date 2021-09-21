from src.net import RNA
from src.network import Network
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 10
TEST_DATASET_SIZE = 10
VALIDATION_DATASET_SIZE = 10
N_EPOCHS = 10
N_NEURONS = 3

FUNCTION = np.sin
LOW = 0 * np.pi
HIGH = 20 * np.pi

def main():
    train_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    validation_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=VALIDATION_DATASET_SIZE, with_noise=False, seed=333)

    net = Network([1, N_NEURONS, 1])
    
    train_loss, test_loss = net.train(train_data.X, train_data.y, N_EPOCHS, 5, lr=0.01, test_X=test_data.X, test_y=test_data.y, cross_validation=True)

    plot_losses(train_loss, test_loss, filename='losses_test-loss-{}.png'.format(test_loss[-1]))
    return

    plot_predictions(validation_data, y_pred, filename='predictions_test-loss-{}.png'.format(test_loss[-1]))
    
if __name__ == '__main__':
    main()