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
        '''Forward propogation of input layer `a`. Returns activation
        layers `a` as list of ndarray'''
        self.activations = []
        for w, b in zip(self.weights, self.biasses):
            a = sigmoid(np.dot(w, a)+b)
            self.activations.append(a)
        return a
        # return [sigmoid(np.dot(w,a)+b) for \
        #                     w, b in zip(self.weights, self.biasses)]

def sigmoid(z):
    '''Computes the sigmoid function'''
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))
