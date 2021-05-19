# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 00:06:28 2021

@author: Aparna
Logistic regression to predict the gender of a person
from a set of input parameters, namely height, weight, and age.
Credits:- https://pub.towardsai.net/logistic-regression-from-scratch-with-only-python-code-9d3ae607e739
"""
import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt
import csv
def read_file(filename):
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        for i in reader:
            dataset.append(i)
    return dataset
dataset = read_file('TrainingSet.csv')
print(dataset)

# Data frame is created under column name Name and Gender 
data_frame = pd.DataFrame(dataset, columns=["Height","Weight","Age", "Gender"])
print(data_frame)

data_frame.Gender[data_frame.Gender == 'M'] = 0
data_frame.Gender[data_frame.Gender == 'W'] = 1
print(data_frame) 

dataset = np.asarray(data_frame)
print(dataset)


def min_max(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_val = [j[i] for j in dataset]
        min_ = min(col_val)
        max_ = max(col_val)
        minmax.append([min_,max_])
    return minmax
minmax = min_max(dataset)
print(minmax)

def normalization(dataset,minmax):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            n = (int(dataset[i][j]) - int(minmax[j][0]))
            d = (int(minmax[j][1]) - int(minmax[j][0]))
            dataset[i][j] = n/d
    return dataset

dataset = normalization(dataset,minmax)
print(dataset)

# Train Test Data
from random import shuffle
def train_test(dataset):
    shuffle(dataset)
    n = int(0.8*len(dataset))
    train_data = dataset[:n]
    test_data =  dataset[n:]
    return train_data,test_data
train_data,test_data = train_test(dataset)
print('Train Data :{} \nTest Data:{}'.format(len(train_data),len(test_data)))

#Accuracy
def accuracy_check(pred,actual):
    c = 0
    for i in range(len(actual)):
        if(pred[i]==actual[i]):
            c+=1
    acc = (c/len(actual))*100
    return acc

# Predict or Hypothesis
def prediction(row,parameters):
    hypothesis = parameters[0]
    for i in range(len(row)-1):
        hypothesis+=row[i]*parameters[i+1]
    return 1 / (1 + math.exp(-hypothesis))

#Cost Function
def cost_function(x,parameters):
    cost = 0
    for row in x:
        pred = prediction(row,parameters)
        y = row[-1]
        #cost+= (y-pred)**2
        cost+= -(y*np.log(pred))+(-(1-y)*np.log(1-pred))
    avg_cost = cost/len(x)
    return avg_cost

#Optimization Technique
def gradient_descent(x,epochs,alpha):
    
    parameters = [0]*len(x[0])
    cost_history = []
    n = len(x)
    
    for i in range(epochs):
        for row in x:
            pred = prediction(row,parameters)
            #for theta 0 partial derivative is different
            parameters[0] = parameters[0]-alpha*(pred-row[-1])
            for j in range(len(row)-1):
                parameters[j+1] = parameters[j+1]-alpha*(pred-row[-1])*row[j]
        cost_history.append(cost_function(x,parameters))
    return cost_history,parameters

#Training and Testing
def algorithm(train_data,test_data):
    
    epochs = 1000
    alpha = 0.001
    cost_history,parameters = gradient_descent(train_data,epochs,alpha)
    predictions = []
    
    for i in test_data:
        pred = prediction(i,parameters)
        predictions.append(round(pred))
    y_actual = [i[-1] for i in test_data]    
    accuracy = accuracy_check(predictions,y_actual)
    
    iterations = [i for i in range(1,epochs+1)]
    plt.plot(iterations,cost_history)
    plt.savefig('cost.png')
    plt.show()
    return accuracy

def combine():
    dataset = dataset = read_file('TrainingSet.csv')
    data_frame = pd.DataFrame(dataset, columns=["Height","Weight","Age", "Gender"])
    data_frame.Gender[data_frame.Gender == 'M'] = 0
    data_frame.Gender[data_frame.Gender == 'W'] = 1
    dataset = np.asarray(data_frame)
    minmax = min_max(dataset)
    dataset = normalization(dataset,minmax)
    train_data,test_data = train_test(dataset)
    accuracy = algorithm(train_data,test_data)
    print(accuracy)

combine()

# Question 2b Starts here



