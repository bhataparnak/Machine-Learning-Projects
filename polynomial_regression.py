# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:54:38 2021

@author: Aparna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_fwf('PolyTrain.txt', header = None, engine='python')
print(df)

X = df.values[:, 0:2]  # get input values from first two columns
y = df.values[:, 2]  # get output values from last coulmn
m = len(y) # Number of training examples

print('Total no of training examples (m) = %s \n' %(m))

def plotCost(numIters,costs):            
            plt.figure()
            plt.plot(np.arange(1, numIters+1), costs, label = r'$J(\theta)$')
            plt.xlabel('Iterations')
            plt.ylabel(r'$J(\theta)$')
            plt.title('Cost vs Iterations of Gradient Descent')
            plt.legend(loc = 'best')
        

def plot_predictedPolyLine(theta):
        """Plot predicted polynomial line using values of theta found
        using normal equation or gradient descent method
        
        Returns
        -----------       
        matploblib figure
        """        
        plt.figure()
        
        line = theta[0] #y-intercept
        print(theta) 
        label_holder = []
        label_holder.append('%.*f' % (2, theta[0]))
              
        
        for i in np.arange(1, len(theta)):            
            line += theta[i] * X ** i 
            label_holder.append(' + ' +'%.*f' % (2, theta[i]) + r'$x^' + str(i) + '$')
        print(label_holder)
        print(line)
        plt.plot(X, line, label = ''.join(label_holder))        
        plt.title('Polynomial Fit: Order ' + str(len(theta)-1))
        plt.xlabel('x')
        plt.ylabel('y') 
        plt.legend(loc = 'best')

def feature_normalize(X):
  """
    Normalizes the features(input variables) in X.

    Parameters
    ----------
    X : n dimensional array (matrix), shape (n_samples, n_features)
        Features(input varibale) to be normalized.

    Returns
    -------
    X_norm : n dimensional array (matrix), shape (n_samples, n_features)
        A normalized version of X.
    mu : n dimensional array (matrix), shape (n_features,)
        The mean value.
    sigma : n dimensional array (matrix), shape (n_features,)
        The standard deviation.
  """
  #Note here we need mean of indivdual column here, hence axis = 0
  mu = np.mean(X, axis = 0)  
  # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
  sigma = np.std(X, axis= 0, ddof = 1)  # Standard deviation (can also use range)
  X_norm = (X - mu)/sigma
  return X_norm, mu, sigma

X, mu, sigma = feature_normalize(X)

print('mu= ', mu)
print('sigma= ', sigma)
print('X_norm= ', X[:m])

mu_test = np.mean(X, axis = 0) # mean
print(mu_test)

sigma_test = np.std(X, axis = 0, ddof = 1) # variance
print(sigma_test)

# use hstack() function from numpy to add column of ones to X feature 
# This will be our final X matrix (feature matrix)

print(X)

def compute_cost(X, y, theta):
  """
  Compute the cost of a particular choice of theta for linear regression.

  Input Parameters
  ----------------
  X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
      m= number of training examples
      n= number of features (including X_0 column of ones)
  y : 1D array of labels/target value for each traing example. dimension(1 x m)

  theta : 1D array of fitting parameters or weights. Dimension (1 x n)

  Output Parameters
  -----------------
  J : Scalar value.
  """
  predictions = X.dot(theta)
  errors = np.subtract(predictions, y)
  sqrErrors = np.square(errors)
  
  J = 1/(2 * m) * errors.T.dot(errors)
  return J

def gradient_descent(X, y, theta, alpha, iterations):
  """
  Compute cost for linear regression.

  Input Parameters
  ----------------
  X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
      m= number of training examples
      n= number of features (including X_0 column of ones)
  y : 1D array of labels/target value for each traing example. dimension(m x 1)
  theta : 1D array of fitting parameters or weights. Dimension (1 x n)
  alpha : Learning rate. Scalar value
  iterations: No of iterations. Scalar value. 

  Output Parameters
  -----------------
  theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
  cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
  """
  cost_history = np.zeros(iterations)
  print(theta)

  for i in range(iterations):
    predictions = X.dot(theta)
   
    errors = np.subtract(predictions, y)
   
    sum_delta = (alpha / m) * X.transpose().dot(errors);
   
    theta = theta - sum_delta;

    cost_history[i] = compute_cost(X, y, theta)  

  return theta, cost_history

# We need theta parameter for every input variable. since we have three input variable including X_0 (column of ones)

# Order 1
theta_order1 = np.zeros(2)
iterations = 100;
alpha = 0.15;
#X = np.hstack((np.ones((m,1)), X))


theta_order1, cost_history00 = gradient_descent(X, y, theta_order1, alpha, iterations)
plot_predictedPolyLine(theta_order1)
plotCost(iterations,cost_history00)

print('Final value of theta =', theta_order1)
print('Values from cost_history =', cost_history00[:m])
print('Values from cost_history =', cost_history00[-m :])

plt.plot(range(1, iterations +1), cost_history00, color ='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")


# order 2

theta_order2 = np.zeros(3)
iterations = 100;
alpha = 0.15;
X = np.hstack((np.ones((m,1)), X))


theta_order2, cost_history0 = gradient_descent(X, y, theta_order2, alpha, iterations)
plot_predictedPolyLine(theta_order2)
plotCost(iterations,cost_history0)

print('Final value of theta =', theta_order2)
print('Values from cost_history =', cost_history0[:m])
print('Values from cost_history =', cost_history0[-m :])

plt.plot(range(1, iterations +1), cost_history0, color ='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")



# order 3

theta_order3 = np.zeros(4)
iterations = 100;
alpha = 0.15;
X = np.hstack((np.ones((m,1)), X))

theta_order3, cost_history1 = gradient_descent(X, y, theta_order3, alpha, iterations)
plot_predictedPolyLine(theta_order3)
plotCost(iterations,cost_history1)

print('Final value of theta =', theta_order3)
print('Values from cost_history =', cost_history1[:m])
print('Values from cost_history =', cost_history1[-m :])

plt.plot(range(1, iterations +1), cost_history1, color ='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")

# Order 4

theta_order4 = np.zeros(5)
iterations = 100;
alpha = 0.15;
X = np.hstack((np.ones((m,1)), X))

theta_order4, cost_history2 = gradient_descent(X, y, theta_order4, alpha, iterations)
plot_predictedPolyLine(theta_order4)
plotCost(iterations,cost_history2)

print('Final value of theta =', theta_order4)
print('Values from cost_history =', cost_history2[:m])
print('Values from cost_history =', cost_history2[-m :])

plt.plot(range(1, iterations +1), cost_history2, color ='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Convergence of gradient descent")



# Testing the model
normalize_test_data1 = ((np.array([6.4432, 9.6309]) - mu) / sigma)
output1 = normalize_test_data1.dot(theta_order1)
print('Predicted output for order 1:', output1)

normalize_test_data2 = ((np.array([6.4432, 9.6309]) - mu) / sigma)
normalize_test_data2 = np.hstack((np.ones(1), normalize_test_data2))
output2 = normalize_test_data2.dot(theta_order2)
print('Predicted output for order 2:', output2)

normalize_test_data3 = ((np.array([6.4432, 9.6309]) - mu) / sigma)
normalize_test_data3 = np.hstack((np.ones(2), normalize_test_data3))
output3 = normalize_test_data3.dot(theta_order3)
print('Predicted output for order 3:', output3)

normalize_test_data4 = ((np.array([6.4432, 9.6309]) - mu) / sigma)
normalize_test_data4 = np.hstack((np.ones(3), normalize_test_data4))
output4 = normalize_test_data4.dot(theta_order4)
print('Predicted output for order 4:', output4)