import numpy as np
from matplotlib import pyplot as plt


def plot_losses(train_loss, test_loss, filename='losses.png'):
    plt.plot(list(range(train_loss.shape[0])), train_loss, label='erro no treino')
    plt.plot(list(range(test_loss.shape[0])), test_loss, label='erro no teste')
    plt.legend()
    plt.xlabel('epocas')
    plt.ylabel('erro')
    plt.title('Erro ao longo das épocas')
    plt.savefig('./plots/' + filename)
    #plt.ylim([0, 1])
    plt.show()


def plot_predictions(data, prediction, filename='predictions.png'):
    # ordena dados no tempo para o plot
    data.X = data.X.reshape(-1)
    data.y = data.y.reshape(-1)
    o = np.argsort(data.X)
    data.X = data.X[o]
    data.y = data.y[o]
    prediction = prediction[o]

    plt.plot(data.X, data.y, label='Real')
    plt.plot(data.X, prediction, label='Predição')
    plt.legend()
    plt.xlabel('x * pi')
    plt.ylabel('sin(x * pi)')
    plt.title('Dado vs Predição')
    plt.savefig('./plots/' + filename)
    plt.show()