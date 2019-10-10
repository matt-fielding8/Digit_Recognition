'''Implementation of a standard neural network algorithm optimised
by stochastic gradient descent. Neural network designed to flexibly predict
hand written digits and sign language images.'''

import numpy as np
from time import perf_counter as timer

class NeuralNetwork:
    def __init__(self, network_size):
        '''Initialises weights and biasses according to the list network_size.
        Each element of network_size corresponds to the number of units in the
        layer.'''
        self.num_layers = len(network_size)
        self.network_size = network_size
        self.predicted_classes = network_size[-1]
        # Weight and biasses scaled down by a factor of 2 to improve forward forward propogation
        # Without scaling, activation layer elements become symmetrically equal to 1
        self.biasses = [np.random.rand(1, i)*0.01 for i in network_size[1:]]
        self.weights = [np.random.rand(i, j)*0.01 for i, j in zip(network_size[:-1], network_size[1:])]

    def forwardProp(self, a):
        '''Forward propogation of input layer `a`. Returns final activation
        layer `a` as ndarray'''
        # Create caches for linear activations
        activations = [a]
        zs = []

        for w, b in zip(self.weights, self.biasses):
            z = np.dot(a, w)+b
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
        b_grads[-1] = np.sum(delta)/delta.shape[0]
        w_grads[-1] = np.dot(activations[-2].T, delta)

        # Propagate backwards through all layers
        for l in range(2, self.num_layers):
            delta = np.dot(delta, self.weights[-l+1].T) * sigmoidPrime(zs[-l])
            b_grads[-l] = np.sum(delta)/delta.shape[0]
            w_grads[-l] = np.array(np.dot(activations[-l-1].T, delta))

        return b_grads, w_grads

    def fit(self, X, y, epochs=1, batch_size=1, shuffle=False, eta=0.01, verbose=False):
        '''Trains the model'''
        steps = X.shape[0] // batch_size
        global_start = timer()

        for epoch in range(epochs):
            start = timer()
            if shuffle:
                seed = np.random.get_state()
                np.random.shuffle(X)
                np.random.set_state(seed)
                np.random.shuffle(y)
            X_batches = [X[i:i+batch_size] for i in range(0,X.shape[0], batch_size)]
            y_batches = [y[i:i+batch_size] for i in range(0,y.shape[0], batch_size)]

            for X_batch, y_batch in zip(X_batches, y_batches):
                self.update_params(X_batch, y_batch, eta)

            if verbose:
                # compute cost of final batch
                a,_ = self.forwardProp(X)
                self.cost = self.logLoss(a[-1], y)
                a_binary = self.predictions(np.array(a[-1]), binary=True)
                self.accuracy = self.predictAccuracy(a_binary, y)
                end = timer()
                print("epoch {}-> Cost: {}, Accuracy: {}, Execution Time: {}"\
                                .format(epoch, self.cost, self.accuracy, (end-start)))

        a, _ = self.forwardProp(X)
        self.cost = self.logLoss(a[-1], y)
        a_binary = self.predictions(np.array(a[-1]), binary=True)
        self.accuracy = self.predictAccuracy(a_binary, y)
        global_end = timer()
        print("Completed ---> Cost: {}, Accuracy: {}, Execution Time: {}"\
              .format(self.cost, self.accuracy, (global_end - global_start)))



    def update_params(self, X, y, eta):
        '''Updates weights and biasses. X and y are batches'''
        # Initialise gradients
        b_grads = [np.zeros(b.shape) for b in self.biasses]
        w_grads = [np.zeros(w.shape) for w in self.weights]


        # Compute backprop to get derivatives
        delta_b_grads, delta_w_grads = self.backProp(X, y)
        # Update gradients
        b_grads = [b+db for b, db in zip(b_grads, delta_b_grads)]
        w_grads = [w+dw for w, dw in zip(w_grads, delta_w_grads)]

        # Update params
        self.weights = [w-(eta/X.shape[0])*dw for w, dw in zip(self.weights, delta_w_grads)]
        self.biasses = [b-(eta/X.shape[0])*db for b, db in zip(self.biasses, delta_b_grads)]


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

        return (-1/m)*np.sum(np.multiply(y,np.log(a))+np.multiply((1-y), np.log(1-a)))

    def predictions(self, a, binary=True):
        '''
        Assigns 1 to class with the maximum probability as 0 to other classes.
        Returns m*k matrix where m is the number of training examples and k
        is the number of classes
        >>> def predictions([[0.1,0.4,0.9,0.3], [0.1,0.7,0.01,0.2]])
        [[0,0,1,0], [0,1,0,0]]
        '''
        if binary:
            return y_binary(np.argmax(a, 1))
        else:
            return np.argmax(a, 1)


    def predictAccuracy(self, a_pred, y, pcnt=False):
        '''
        Compares predicted values with actual values.
        Returns accuracy as a float or string.
        '''

        if pcnt:
            accuracy = np.mean([a == y for a, y in zip(a_pred, y)])
            return "{:.2f}%".format(accuracy*100)
        else:
            return np.mean([a == y for a, y in zip(a_pred, y)])



def sigmoid(z):
    '''Computes the sigmoid function'''
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))


def y_binary(y, min_class=0):
    '''
    Converts numerical class values to binary vectors for multiclass problems.
    Returns m x k matrix where m is the number of examples and k is number of classes.
    >>> def y_binary([1,4,3])
    [[1,0,0,0],[0,0,0,1],[0,0,1,0]]
    '''
    # Class range starts at zero
    if not min_class:
        matrix = np.zeros((y.shape[0], y.max() + 1))
        for i, j in enumerate(y):
            matrix[i][j] += 1
    # Class range starts at one
    else:
        matrix = np.zeros((y.shape[0], y.max()))
        for i, j in enumerate(y):
            matrix[i][j - 1] += 1

    return np.array(matrix)
