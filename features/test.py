import numpy as np
import pandas as pd
from features import model as mod

df = pd.read_csv('../data/train.csv', nrows=100)
print(df.shape)
print(df.head())

X = df.iloc[1,1:].values.reshape(784,1)
temp = df.iloc[1,0]
y = np.zeros(10)
y[temp] = 1
y = y.reshape(10,1)

# print(y.shape)

# print(X)
# print(X.shape)

net = mod.NeuralNetwork([784,16,16,10])

b = net.backProp(X, y)

print("Bias length", [i.shape for i in net.biasses])
print("Weights length", [i.shape for i in net.weights])

print(b)

# a = net.forwardProp(X)


# print(net.activations)

# print(a.shape)
# print([i.shape for i in net.activations])
# print(a)

# a = np.array([[0,1],[2,3]])
# b = np.array([[1],[1]])

# print(a)
# print(b)
# print(a+b)

# print((np.dot(net.weights[0], X).T + net.biasses[0]))
# print(np.dot(net.weights[0], X).shape)
# print(net.biasses[0].shape)
#
# print(list(reversed(enumerate(np.ndarray([4,1])))))