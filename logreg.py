# -*- coding: utf-8 -*-
"""
This program implements logistic regression using
1) Gradient Descent
2) Stochastic Gradient Descent 
3) Newton's Method

Author: Ankit Bansal
Email: arbansal@umich.edu
"""
import numpy as np
import matplotlib.pyplot as plt

def readData(fileX, fileY):
    """
    Reads data from files and returns as numpy arrays
    Input: file locations
    Output: X,Y, N number of data points
    """
    X = []
    Y = [] 
    with open(fileX,'r') as f_x, open(fileY,'r') as f_y:
        for line in f_x:
            temp = line.split()
            temp = np.append(1,temp)
            X += [temp]
        for line in f_y:
            Y += [line.split()]       

    X = np.asarray(X, dtype=np.double)
    N = X.shape[0]
    Y = np.asarray(Y, dtype=np.double)
    
    return N,X,Y

def sig(x):
    """
    Returns the sigmoid/logistic function of input
    """
    return 1/(1 + np.exp(-x))

def Loss(W):
    """
    Computes the negative log likelihood function for the parameter W
    Input: Weight matrix W
    Output: loss = -log(l(W))
    """
    loss = 0
    for i in range(N):
        loss += -Y[i]*np.log(sig(np.inner(W,X[i]))) - (1 - Y[i])*np.log(1 - sig(np.inner(W,X[i])))
    return loss

def GD(X,Y,eta,tol):
    """
    Performs Gradient Descent 
    Input: X,Y,eta (learning rate), tol (tolerance)
    Output: Weight Matrix W, error
    """
    W = np.array([0.0,0.0,0.0])
    error = 10   
    steps = 0
    while(error > tol):
        l1 = Loss(W)
        steps += 1
        deltaW = 0 
        deltaW += gradL(X,Y,W)
        W = W - eta*deltaW  
        l2 = Loss(W)
        error = np.abs(l2 - l1)
    return W,error,steps    

def SGD(X,Y,eta_o,tol):
    """
    Performs Stochastic Gradient Descent 
    Input: X,Y,eta_o (learning rate), tol (tolerance)
    Output: Weight Matrix W, error
    """
    W = np.array([0.0,0.0,0.0])
    error = 10 
    steps = 0
    
    while(error > tol):
        steps += 1
        i = np.random.randint(N)
        l1 = Loss(W)
        eta = eta_o/(1 + eta_o*steps)**(3/4)   
        W = W - eta*(sig(np.dot(W.T,X[i])) - Y[i])*X[i]  
        l2 = Loss(W)
        error = np.abs(l2 - l1)
    return W,error,steps 
def gradL(X,Y,W):
    """
    Computes the gradient of L(W)
    """
    Y = np.squeeze(Y)
    D = sig(np.inner(W,X)) - Y      
    G = np.inner(X.T,D)
    return G

def hessian(X,N,W):
    """
    Computes the Hessian matrix
    """
    h_w = []
    for i in range(N):    
        h_w += [sig(np.inner(W,X[i]))*(1 - sig(np.inner(W,X[i])))] 
    
    h_w = np.asarray(h_w)
    B = np.diag(h_w)
    H = np.dot(X.T,np.dot(B,X))
    return H
    
def Newton(X,Y,tol):
    """
    Performs Newton's method to update the weight matrix
    Input: X,Y, tol(tolerance)
    Output: Weight Matrix W, error
    """
    W = np.array([0.0,0.0,0.0])
    error = 10
    steps = 0
    while(error > tol):
        l1 = Loss(W)
        steps += 1
        for i in range(N):
            H = hessian(X,N,W)
            G = gradL(X,Y,W)
            W = W - np.dot(np.linalg.pinv(H),G)
        l2 = Loss(W) 
        error = np.abs(l2 - l1)
    return W,error,steps  

if __name__ == '__main__':  
    
    N,X,Y = readData('q1x.dat', 'q1y.dat')   
    W, error,steps = GD(X,Y,0.001,10**(-8))
    print("\nGradient Descent\n", W, "\nError: ", error, " Steps: ", steps)
    
    W, error, steps = SGD(X,Y,1, 10**(-8))
    print("\nStochastic Gradient Descent\n", W, "\nError: ", error, " Steps: ", steps)
    
    W, error,steps = Newton(X,Y,10**(-8))
    print("\nNewton's\n", W, "\nError: ", error, " Steps: ", steps)

    X1_0 = []
    X2_0 = []
    X1_1 = []
    X2_1 = []
    for i,t in enumerate(Y):
        if(t == 0):
            
            X1_0 += [X[:,1][i]]
            X2_0 += [X[:,2][i]]
            print(X1_0)
        else:
            X1_1 += [X[:,1][i]]
            X2_1 += [X[:,2][i]]
            
    print(len(X1_0))        
    plt.scatter(X1_0,X2_0)
    plt.scatter(X1_1,X2_1)
    X1 = X[:,1]
    T1 = - W[1]/W[2]*X1
    T2 = [-W[0]/W[2] for i in range(X.shape[0])]
    Y1 = T2 + T1        
    plt.plot(X1,Y1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Binary Classification decision boundary')
        
