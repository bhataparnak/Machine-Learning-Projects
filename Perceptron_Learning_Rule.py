# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:54:28 2021

@author: Aparna
"""

import numpy as np

X = np.array((
    [1, 1, 1],
    [1, 1, -1],
    [1, 0, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1]))
indices_one = X == -1
X[indices_one] = 0 # replacing 1s with 0s
print(X)

y = np.array(([1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
              [-1]))

indices_one = y == -1
y[indices_one] = 0 # replacing 1s with 0s
print(y)

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 3
    self.outputSize = 1
   

    #weights
    self.W = np.random.randn(self.inputSize, self.outputSize) # (8x4) weight matrix from input to hidden layer
    
  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, (self.W+1)) # dot product of X (input) and first set of 8x3 weights + bias
    o = self.sigmoid(self.z) # activation function
    return o 

  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return self.sigmoid(s) * (1 - self.sigmoid(s))

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
    self.W += X.T.dot(self.o_delta) # adjusting set (input --> output) weights
    
  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

NN = Neural_Network()
for i in range(10): # trains the NN 10 times
  print ("Input: \n" + str(X))
  print ("Actual Output: \n" + str(y))
  print ("Predicted Output: \n" + str(NN.forward(X)))
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  NN.train(X, y)