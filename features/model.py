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
        activations = [a]
        zs = []

        for w, b in zip(self.weights, self.biasses):
            z = (np.dot(w,a)+b)
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        return activations, zs


    def backProp(self, X, y):
        '''Back propogation to get gradients'''
        # Initialise list of gradient vectors
        b_grads = [np.zeros(b.shape) for b in self.biasses]
        w_grads = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activations, zs = self.forwardProp(X)

        # Initialise delta
        delta = (activations[-1] - y) * sigmoidPrime(zs[-1])
        b_grads[-1] = delta
        w_grads[-1] = np.dot(delta, activations[-2].T)

        # Propagate backwards through all layers
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoidPrime(zs[-l])
            b_grads[-l] = np.array(delta)
            w_grads[-l] = np.array(np.dot(delta, activations[-l].T))

        return b_grads, w_grads


    def MSE(self, a, y):
        '''Calculates mean squared error for predictions `a` and actual values
        `y`.'''
        # Compute batch size
        m = y.shape[0]
        return np.sum((y-a)**2)/(2*m)

    def logLoss(self, a, y):
        '''Calculates logarithmic loss for predictions `a` and actual values
        `y`. Equivalent to categorical_crossentropy in Keras'''
        # Compute batch size
        m = y.shape[0]
        print(m)
        return (-1/m)*np.sum(y*np.log(a)+(1-y)*np.log(1-a))



def sigmoid(z):
    '''Computes the sigmoid function'''
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))
