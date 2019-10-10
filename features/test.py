import numpy as np
import pandas as pd
from features import model as mod

df = pd.read_csv('../data/train.csv', nrows=10000)
print(df.shape)
print(df.head())

X = df.iloc[:,1:].values
X = X/255
print(X.max())
y = df.iloc[:,0]
y = mod.y_binary(y)

net = mod.NeuralNetwork([784,150,150,150,150,10])

net.fit(X,y,epochs=10,batch_size=100)

