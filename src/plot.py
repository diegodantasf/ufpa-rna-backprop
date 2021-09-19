import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def plot_losses(train_loss, test_loss):
  plt.plot(list(range(len(train_loss))), train_loss, label='erro no treino')
  plt.plot(list(range(len(test_loss))), test_loss, label='erro no teste')
  plt.legend()
  plt.title('Erro ao longo das épocas')
  plt.savefig('./plots/losses_{}.png'.format(datetime.now().__str__().replace(':', '-').replace(' ', '_')))
  plt.show()


def plot_predictions(data, prediction):
  # ordena dados no tempo para o plot
  o = np.argsort(data.X)
  data.X = data.X[o]
  data.y = data.y[o]
  prediction = prediction[o]


  plt.plot(data.X, data.y, label='Real')
  plt.plot(data.X, prediction, label='Predição')
  plt.legend()
  plt.title('Dado vs Predição')
  plt.savefig('./plots/predictions_{}.png'.format(datetime.now().__str__().replace(':', '-').replace(' ', '_')))
  plt.show()