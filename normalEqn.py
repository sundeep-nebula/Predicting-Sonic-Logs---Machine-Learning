# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 07:50:48 2018

@author: sharms7
"""
import os
import numpy as np

def normalEqn(X,y):
    t1=np.linalg.pinv(np.dot(np.transpose(X),X))
    t2=np.dot(np.transpose(X),y)
    theta=np.dot(t1,t2)
    return theta