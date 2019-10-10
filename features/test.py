import numpy as np
import pandas as pd
from features import model as mod

df = pd.read_csv('../data/train.csv', nrows=100)
print(df.shape)
print(df.head())

X = df.iloc[:,1:].values
X = X/255
print(X.max())
y = df.iloc[:,0]
y = mod.y_binary(y)

net = mod.NeuralNetwork([784,784,10])

net.fit(X,y,epochs=5,batch_size=1)

a, zs = net.forwardProp(X)

print("a", a)
print("zs", zs)
print("sigs", [mod.sigmoid(z) for z in zs])

