import numpy as np

class RNA(object):
    def __init__(self, n1, seed=0):
        rng = np.random.default_rng(seed)
        self.n1 = n1
        self.w1 = rng.random(n1)
        self.w2 = rng.random(n1)
        self.b1 = rng.random(n1)

    def compute_gradients(self, X, y):
        n = X.shape[0]
        self.gw1 = np.zeros(self.n1)
        self.gw2 = np.zeros(self.n1)
        self.gb1 = np.zeros(self.n1)

        self.errors = np.zeros(n)
        for i in range(n):
            self.errors[i] = self.forward(X[i]) - y[i]

        for i in range(self.n1):
            ans = 0
            for j in range(n):
                ans += self.errors[j] * self.w2[i] * (1 - self.g(self.w1[i] * X[j] + self.b1[i])**2) * X[j]
            self.gw1[i] = 2 / n * ans
        
        for i in range(self.n1):
            ans = 0
            for j in range(n):
                ans += self.errors[j] * self.g(self.w1[i] * X[j] + self.b1[i])
            self.gw2[i] = 2 / n * ans

        for i in range(self.n1):
            ans = 0
            for j in range(n):
                ans += self.errors[j] * self.w2[i] * (1 - self.g(self.w1[i] * X[j] + self.b1[i])**2)
            self.gb1[i] = 2 / n * ans

    def train(self, X, y, num_epochs = 10, lr=0.001):
        n = X.shape[0]
        loss = []
        for epoch in range(num_epochs):
            self.compute_gradients(X, y)
            for i in range(self.n1):
                self.w1[i] = self.w1[i] - lr * self.gw1[i]
            for i in range(self.n1):
                self.w2[i] = self.w2[i] - lr * self.gw2[i]
            for i in range(self.n1):
                self.b1[i] = self.b1[i] - lr * self.gb1[i]
            
            cur_loss = self.loss(X, y)
            loss.append(cur_loss)

            print(f'epoch {epoch + 1} loss {cur_loss}')

        return loss
    
    def loss(self, X, y):
        ans = 0
        for i in range(len(X)):
            ans += (self.forward(X[i]) - y[i])**2
        ans /= len(X)
        return ans

    def forward(self, x):
        ans = 0
        for i in range(self.n1):
            ans += self.w2[i] * self.g(self.w1[i] * x + self.b1[i])
        return ans

    @staticmethod
    def g(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


