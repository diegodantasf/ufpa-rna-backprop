from src.net import RNA
from src.dataset import Dataset

import matplotlib.pyplot as plt
import numpy as np

def main():
    data = Dataset(np.sin, lo=0, hi=2 * np.pi, n=1000, with_noise=False)
    net = RNA(5)

    losses = net.train(data.X, data.y, num_epochs=500, lr=0.01)

    plt.plot(list(range(len(losses))), losses)
    plt.title('Loss over epochs')
    plt.show()

    o = np.argsort(data.X)
    data.X = data.X[o]
    data.y = data.y[o]

    y_pred = np.zeros(1000)
    for i in range(len(y_pred)):
        y_pred[i] = net.forward(data.X[i])

    plt.plot(data.X, data.y, label='Real')
    plt.plot(data.X, y_pred, label='Prediction')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()