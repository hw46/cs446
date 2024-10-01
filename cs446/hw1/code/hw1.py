import numpy as np
import hw1_utils as utils
from hw1_utils import load_reg_data
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    x=np.hstack((np.ones((X.shape[0],1)),X))
    w=np.zeros(x.shape[1])
    for i in range(num_iter):
        w=w-lrate*(1/x.shape[0])*x.T.dot(x.dot(w)-Y)
    return w

def linear_normal(X,Y):
    x=np.hstack((np.ones((X.shape[0],1)),X))
    w=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(Y)
    return w

def plot_linear():
    X,Y=utils.load_reg_data()
    w=linear_normal(X,Y)
    X_bias=np.hstack((np.ones((X.shape[0],1)),X))
    predictions=X_bias.dot(w)
    plt.scatter(X[:,0],Y,color='blue',label='Data Points')
    sorted_indices=X[:,0].argsort()
    X_sorted=X[sorted_indices]
    predictions_sorted=predictions[sorted_indices]
    plt.plot(X_sorted[:,0],predictions_sorted,color='red',label='Regression Line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()
    
plot_linear()