from src.net import RNA
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 100
TEST_DATASET_SIZE = 100
VALIDATION_DATASET_SIZE = 100
N_EPOCHS = 1000
N_NEURONS = 32

FUNCTION = np.sin
LOW = -10 * np.pi
HIGH = 10 * np.pi

def main():
    train_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    validation_data = Dataset(FUNCTION, lo=LOW, hi=HIGH, n=VALIDATION_DATASET_SIZE, with_noise=False, seed=333)

    net = RNA(N_NEURONS)

    [train_loss, test_loss] = net.train(train_data.X, train_data.y, test_data.X, test_data.y, num_epochs=N_EPOCHS, lr=0.0001)
    y_pred = net.predict(validation_data.X)

    plot_losses(train_loss, test_loss, filename='losses_test-loss-{}.png'.format(test_loss[-1]))

    plot_predictions(validation_data, y_pred, filename='predictions_test-loss-{}.png'.format(test_loss[-1]))
    
if __name__ == '__main__':
    main()