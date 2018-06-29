# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 07:51:06 2018

@author: sharms7
"""
import os
import numpy as np

def featureNormalize(X):
    X_norm=X
    mu=np.mean(X)
    sigma=np.std(X)
    X=(X-mu)/sigma
    X_norm=X
    return X_norm,mu,sigma