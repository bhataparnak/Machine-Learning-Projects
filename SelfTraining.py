# import the class
import numpy as np
from matplotlib import pyplot as plt
import math
from numpy import array, log, e
from mpl_toolkits.mplot3d import Axes3D

#Given training data with labels
trainX=np.array([170,57,32,190,95,28,150,45,35,168,65,29,175,78,26,185,90,32,171,65,28,155,48,31,165,60,27]).reshape(9,3)
trainY=np.array([0,1,0,1,1,1,0,0,0])
#Given unlabeled training data
trainX_unlabeled=np.array([182,80,30,175,69,28,178,80,27,160,50,31,170,72,30,152,45,29,177,79,28,171,62,27,185,90,30,181,83,28,168,59,24,
               158,45,28,178,82,28,165,55,30,162,58,28,180,80,29,173,75,28,172,65,27,160,51,29,178,77,28,182,84,27,175,67,28,
               163,50,27,177,80,30,170,65,28]).reshape(25,3)
#Test Data
testX=np.array([169,58,30,185,90,29,148,40,31,177,80,29,170,62,27,172,72,30,175,68,27,178,80,29]).reshape(8,3)
testY=np.array([0,1,0,1,0,1,0,1])

#We consider Woman as 0 and man as 1
W=0
M=1
k=1
#Sigmoid function
def sigmoid(z):
    return 1 / (1 + e**(-z))

for epoch in range(0,25,k):
    #print(epoch)
    if(epoch>25):
        break
    trainX=trainX.reshape(epoch+9,3)
    #print(trainX)
    m=trainX.shape[0]
    N=trainX.shape[0]
    a = np.array(trainX)
    b = a.transpose()
    c = np.ones(N, dtype = int)
    alpha=0.0001
    X=np.vstack((c, b))

    theta=np.zeros(4)
    for x in range(1000):
        h=np.dot(theta.transpose(), X)
        g=sigmoid(h)  
        predict_1 =np.dot(trainY, log(g))
        predict_0 = np.dot((1-trainY), log(1-g))
        cost=-(predict_1 + predict_0) / N
        #print(cost)
        theta = theta+alpha*np.dot(X, (trainY-g))/m    
    print("Value of theta in iteration",epoch+1,":\n",theta)

    trainX_unlabeled=trainX_unlabeled
    #print(trainX_unlabeled)

    a = np.array(trainX_unlabeled)
    b = a.transpose()
    c = np.ones(len(trainX_unlabeled), dtype = int)
    testX_unlabeled=np.vstack((c, b))
    temp=np.dot(theta.transpose(), testX_unlabeled)
    predict=sigmoid(temp)
    #Constructing the confidence list
    conf=[]
    for p in predict:
        if (p<0.5):
            conf.append(1-p)
        else:
            conf.append(p)
    print("Confidence values from the unlabeled dataset:")
    print(conf)

    #Sorting the confidence list in descending order
    #taking the indices of the unlabeled data having the top k values
    indices=np.argsort(-np.array(conf))[:k]
    print(trainX_unlabeled)
    for index in indices:
        trainX=np.append(trainX,trainX_unlabeled[index])
        trainX_unlabeled=np.delete(trainX_unlabeled, index, 0)
        if predict[index]<0.5:
            trainY=np.append(trainY,W)
        else:
            trainY=np.append(trainY,M)
    print("########################################################################################\n")
#Testing the test data after self-training
a = np.array(testX)
b = a.transpose()
c = np.ones(len(testX), dtype = int)
test_X=np.vstack((c, b))
temp=np.dot(theta.transpose(), test_X)
predict=sigmoid(temp)
pred=[]
for x in predict:
    if x>=0.5:
        pred.append(1)
    else:
        pred.append(0)

print("Prediction after self-training:")
print(pred)

print("Accuracy:",sum(1 for x,y in zip(testY,pred) if x == y) / len(a)*100,"%")

SSE=np.sum(np.square(testY-pred))
print("SSE for Test Data:",SSE)


