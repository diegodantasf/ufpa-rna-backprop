from src.net import RNA
from src.dataset import Dataset
from src.plot import *

import matplotlib.pyplot as plt
import numpy as np

TRAIN_DATASET_SIZE = 1000
TEST_DATASET_SIZE = 1000
N_EPOCHS = 50
N_NEURONS = 5

def main():
    train_data = Dataset(np.sin, lo=0, hi=2 * np.pi, n=TRAIN_DATASET_SIZE, with_noise=False, seed=1337)
    test_data = Dataset(np.sin, lo=0, hi=2 * np.pi, n=TEST_DATASET_SIZE, with_noise=False, seed=42)

    net = RNA(N_NEURONS)

    [train_loss, test_loss] = net.train(train_data.X, train_data.y, test_data.X, test_data.y, num_epochs=N_EPOCHS, lr=0.01)

    validation_data = Dataset(np.sin, lo=0, hi=2 * np.pi, n=TEST_DATASET_SIZE, with_noise=False, seed=42)
    y_pred = net.predict(validation_data.X)

    plot_losses(train_loss, test_loss)

    plot_predictions(validation_data, y_pred)
    
if __name__ == '__main__':
    main()