import numpy as np

class Dataset(object):
    def __init__(self, func, lo, hi, n=1000, with_noise=True, seed=1337):
        self.rng = np.random.default_rng(1)
        self.X = self.rng.uniform(lo, hi, n)
        self.X = np.reshape(self.X, [self.X.shape[0], 1])
        self.y = func(self.X)
        self.y = np.reshape(self.y, [self.y.shape[0]])
        if with_noise:
            self.y += self.rng.normal(0, 0.01, n)

    def __len__(self):
        return len(self.X)
     
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

