# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 07:29:48 2018

@author: sharms7

This is a code to calculate gradient descent. 
usage: [theta,J_history]= gradientDescent(X,y,theta,alpha,num_iters)

## INPUTS ##
X is a matrix that M x (N+1) matrix where M is the number of training examples
and N is the number of features. N+1 because you will have to add the intercept
coefficient as all 1's in the zeroth column.
y is a Mx1 vector with the dataset to be predicted.
theta is a (N+1)x1 vector which contain initial theta values to begin gradient
descent
alpha is the learning rate.
num_iters is the number of iterations for gradient descent calculation

## OUTPUTS ##
theta is a (N+1)x1 vector of theta values that is obtained after gradient 
descent has finished running to the desired number of iterations. 

J_history is history of cost function values at each iteration of gradient
descent. There is another script called "computeCostMulti" that calculates 
cost function values given X,y and initian theta values.    
 
"""
import os
import sys
import numpy as np
import costfunc as cc



def gradientDescent(X, y, theta, alpha, num_iters):

    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        a=np.matmul(np.transpose(X),X)
        b= np.dot(np.transpose(X),y)
        delta=(1./m)*(np.dot(a,theta)-b)
        theta=theta-np.dot(alpha,delta)
        
        J_history[i] = cc(X, y, theta)
        
        # print(J_history[i])

    return theta,J_history
