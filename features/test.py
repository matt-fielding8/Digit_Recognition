import numpy as np
import pandas as pd
from features import model as mod

df = pd.read_csv('../data/train.csv', nrows=100)
print(df.shape)
print(df.head())

X = df.iloc[1,1:]

# print(X)
print(X.shape)

net = mod.NeuralNetwork([784,16,16,10])

print("Bias length", [i.shape for i in net.biasses])
print("Weights length", [i.shape for i in net.weights])

a = net.forwardProp(X)

print(net.activations)

print(a.shape)

