import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time
import pickle


#activation function Zl -> Al
def relu(Z):
    return np.maximum(Z, 0)
def relu_derivative(Z):
    return (Z > 0).astype(float)

# softmax function (activation for last layer) ZL -> AL
def softmax(Z):
    C = np.max(Z, axis=0)
    e = np.exp(Z - C)
    return e / np.sum(e, axis=0)

# as a loss function I have used CCE - categorical cross entropy Li = -Sum(Col)[ Yk * log(A_k^l) ]
# its never implemented because we use only derivative which is dL/dAL, but it is not needed to calculate it because
# first delta in backpropagation is dL/dZL = dL/dAL * dAL/dZL = AL - Y


class NeuralNetwork:
    # initialize network
    def __init__(self, layer):
        # NeuralNetwork([784, 128, 10])
        self.A = {} # A[0] = (784, m)
        self.Z = {} # Z[0] - doesnt exist, Z[1] = (128, m)
        self.grad = {} # grad[l][0] = ▽(Wl)C  grad[l][1] = ▽(bl)C - matrices of gradient grad[l][0][i][j] = dC/dW(l-1->l)(j->i)
        self.layers_num = len(layer)
        self.W = {}
        self.b = {}
        for l in range(1, self.layers_num):
            n_curr, n_prev = layer[l], layer[l-1]
            self.W[l] = np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            # He Initialization - multiplying by np.sqrt(2 / n_prev)
            # Var(z) = nVar(W) * Var(a), so sqrt(2 / n_prev) makes this multipliers equals 1
            # as a result gradient variance will not be exponentially growing (causing grad-des algorithm to fail)
            # and gradient variance will not vanish to zero, causing grad-des algorithm to fail (steps will be too small)
            self.b[l] = np.zeros([n_curr, 1])
            self.grad[l] = [{}, {}]

    #forward pass
    def forward(self, X):
        self.A[0] = X
        for l in range(1, self.layers_num):
            self.Z[l] = self.W[l] @ self.A[l-1] + self.b[l]
            self.A[l] = relu(self.Z[l]) if l != self.layers_num - 1 else softmax(self.Z[l]) #activation ReLU or SoftMax
        return self.A[self.layers_num-1]

    #backpropagation
    def backprop(self, X, Y):
        m = Y.shape[1] # Y = (10, m) so batch size is deduced automatically
        delta = {} # dynamical programming array of error signals dL/dZl
        # calculated using chain rule
        # ΔL = dL/dZL = dL/dAL * dAL/dZL = ... = AL - Y
        # Δl = ((Wl+1)T * Δl+1) * ReLU'(Zl) - Hadamard product
        delta[self.layers_num - 1] = self.A[self.layers_num - 1] - Y
        for l in range(self.layers_num - 2, 0, -1):
            delta[l] = self.W[l+1].T @ delta[l+1] * relu_derivative(self.Z[l])

        # gradient calculation using error signals (sigmas) dL/dW =chain rule= dZ/dW * dL/dZ=delta
        # ▽WL = 1/m * Δl * (Al-1)T
        # ▽bL = 1/m * Sum(1->m)[ Δ_(.,k)^l ]
        for l in range(1, self.layers_num):
            self.grad[l][0] = 1/m * delta[l] @ self.A[l-1].T
            self.grad[l][1] = 1 / m * np.sum(delta[l], axis=1, keepdims=True)

        return self.grad

    #gradient descent
    def update(self, eta): # eta η - learning rate
        for l in range(1, self.layers_num):
            self.W[l] -= eta * self.grad[l][0]
            self.b[l] -= eta *self.grad[l][1]

    #learning full-batch
    def learn_full_batch(self, X, Y, eta, epochs):
        for epoch in range(epochs):
            self.forward(X) # forward pass fills all A and Z
            self.backprop(X, Y) # backpropagation uses A Z to find deltas and gradient of cost-function
            self.update(eta) # update uses gradient descent for findind local minimum of cost function

    # learning mini-batches
    def learn_mini_batch(self, X, Y, eta, epochs, batch_size):
        for epoch in range(epochs):
            total_m = X.shape[1]
            perm = np.random.permutation(total_m)
            # performs random but the same permutation of X, Y
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]


                # makes bathces and iterates throw them performing gradient descent every iteration
            for batch in range(0, total_m, batch_size):
                X_batch = X_shuffled[:, batch : batch + batch_size]
                Y_batch = Y_shuffled[:, batch : batch + batch_size]
                self.forward(X_batch)
                self.backprop(X_batch, Y_batch)
                self.update(eta)

    # saves trained model to file
    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump({"W": self.W, "b": self.b}, f)

    # loads trained model from .pkl file
    def load_model(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W = data["W"]
        self.b = data["b"]

    def predict(self, X):
        # gives an assumption of network on given data X
        # X = (784, m) -> (10, m) -> (m, ) - prediction of number of neural network
        return np.argmax(self.forward(X), axis=0)

# one_hot (10, 1) vector representation of number Y = (10, m)
def one_hot(Y_raw, num_results):
    Y = np.zeros((num_results, Y_raw.size), dtype=np.int8)
    for l in range(Y_raw.size):
        Y[Y_raw[l]][l] = 1
    return Y

# gives arithmetic average of correct answers, X = (784, m) -> (m, ) ?= (m, ) = Y
# so k/m - where k is number of correct answers
def accuracy(model, X, Y):
    return np.mean(model.predict(X) == Y)

def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  # X = (70000, 784) y = (70000, )
    X = X / 255.0  # normalize pixels, by default mnist gives 0-255 values of every pixel

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # splits all test pile on 2 parts 80%-train data 20%-test data to check accuracy
    X_train, X_test = X_train.T, X_test.T
    Y_train = one_hot(y_train, 10)  # makes one_hot matrix (10, m) from (m, )

    return X_train, X_test, Y_train, y_test

def main():

    X_train, X_test, Y_train, y_test = load_data()

    net = NeuralNetwork([784, 128, 10])
    # X = A0 = 784, Z1 = A1 = 128, Z2 = A2 = Y = 10
    # W[0]=0,  W[1] = (128, 784) b[1] = (128, 1) W[2] = (10, 128) b[2] = (10, 1)
    start = time.perf_counter()
    net.learn_mini_batch(X_train, Y_train, eta=0.5, epochs=20, batch_size=64) # it trains on the same pile of data epochs times
    end = time.perf_counter()
    elapsed = end - start
    print(f"took {elapsed:.2f} seconds\nachieved accuracy: {accuracy(net, X_test, y_test):.2f}")
    # net.save_model("model_mini_b_784_128_10.pkl")


if __name__ == "__main__":
    main()


