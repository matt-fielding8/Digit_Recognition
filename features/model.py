'''Implementation of a standard neural network algorithm optimised
by stochastic gradient descent. Neural network designed to flexibly predict
hand written digits and sign language images.'''

import numpy as np

class NeuralNetwork:
    def __init__(self, network_size):
        '''Initialises weights and biasses according to the list network_size.
        Each element of network_size corresponds to the number of units in the
        layer.'''
        self.num_layers = len(network_size)
        self.network_size = network_size
        self.predicted_classes = network_size[-1]

        self.biasses = [np.random.rand(i,1) for i in network_size[1:]]
        self.weights = [np.random.rand(i, j) for j, i in
                            zip(network_size[:-1], network_size[1:])]

    def forwardProp(self, a):
        '''Forward propogation of input layer `a`. Returns final activation
        layer `a` as ndarray'''
        self.activations = []
        for w, b in zip(self.weights, self.biasses):
            a = sigmoid(np.matmul(w, a)+b)
            self.activations.append(a)
        return a


    def backProp(self, X, y):
        '''Back propogation to get gradients'''
        # Initialise list of delta vectors
        deltas = [np.zeros([i,1]) for i in self.network_size]
        grads = [np.zeros([i,1]) for i in self.network_size]

        print("delts", [i.shape for i in deltas])

        a = X
        # List to store activation layers
        activations = [X]
        # List to store z vectors
        zs = []

        # Forward pass
        for w, b in zip(self.weights, self.biasses):
            z = (np.dot(w,a)+b)
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        deltas[-1] = (a - y)

        print("zs", [i.shape for i in zs])
        print("weights", [i.shape for i in self.weights])

        for l in range(2, self.num_layers):
            deltas[-l] = np.dot(self.weights[-l+1].T, deltas[-l+1])
            grads[-l] = np.dot(deltas[-l+1], activations[-l].T)*\
                            sigmoidPrime(zs[-l+1])

        return grads


    def MSE(self, a, y):
        '''Calculates mean squared error for predictions `a` and actual values
        `y`.'''
        m = len(a)
        return np.sum((y-a)**2)/(2*m)

def sigmoid(z):
    '''Computes the sigmoid function'''
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))
