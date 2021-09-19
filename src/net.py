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
        self.errors = np.subtract(self.forward(X), y)

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

    def train(self, train_X, train_y, test_X, test_y, num_epochs = 10, lr=0.09):
        n = train_X.shape[0]
        train_loss = []
        test_loss = []
        for epoch in range(num_epochs):
            self.compute_gradients(train_X, train_y)

            self.w1 = np.subtract(self.w1, np.multiply(lr, self.gw1))

            self.w2 = np.subtract(self.w2, np.multiply(lr, self.gw2))

            self.b1 = np.subtract(self.b1, np.multiply(lr, self.gb1))
            
            cur_train_loss = np.mean(self.loss_fuction(train_X, train_y))
            train_loss.append(cur_train_loss)

            cur_test_loss = np.mean(self.loss_fuction(test_X, test_y))
            test_loss.append(cur_test_loss)

            print(f'epoch {epoch + 1} train_loss {cur_train_loss}')
            print(f'epoch {epoch + 1} test_loss {cur_test_loss}')

        return [train_loss, test_loss]
    
    def test(self, X, y):
        return self.loss_fuction(X, y)
    
    def predict(self, X):
        return self.forward(X)
    
    def loss_fuction(self, X, y):
        return np.subtract(self.forward(X), y)**2

    def forward(self, X):
        results = np.zeros(np.shape(X))
        for i in range(len(X)):
            results[i] = np.dot(self.w2, self.g(np.add(np.multiply(self.w1, X[i]), self.b1)))
        
        return results

    @staticmethod
    def g(x):
        return np.tanh(x)


