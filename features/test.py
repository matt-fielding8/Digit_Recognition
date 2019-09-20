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

net = mod.NeuralNetwork([784,16,16,10])

a = net.forwardProp(X)
b = net.backProp(X, y)

print("Bias length", [i.shape for i in net.biasses])
print("Weights length", [i.shape for i in net.weights])
print("Grads", [i.shape for i in b])

