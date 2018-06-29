# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 07:30:21 2018

@author: sharms7
"""
import os
import numpy as np
def computeCost(X,y,theta):
    m = len(y)
    s = np.power(( X.dot(theta) - np.transpose([y]) ), 2)
    J = (1.0/(2*m)) * s.sum( axis = 0 )
    return J

def computeCostMulti(X,y,theta):
    m=len(y);
    predictions=np.dot(X,theta)
    sqerrors=np.power(predictions-y,2)
    J=(1./(2*m))*np.sum(sqerrors)
    return J