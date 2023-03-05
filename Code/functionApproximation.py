#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:38:53 2023

@author: gouthamgopalakrishna
"""
import numpy as np
import matplotlib.pyplot as plt
#functional approximation
x = np.linspace(-2,1.5,1000)
y = x**3 + x**2 - x -1
yhat = -np.maximum(0,-7.7-5*x) - np.maximum(0,-1.3-1.2*x) -\
         np.maximum(0,1+1.2*x) + np.maximum(0,-0.2+1.2*x) +\
         np.maximum(0,-1.1+2*x) + np.maximum(0,-5+5*x) 

plt.figure()
plt.plot(x,y,label='actual')
plt.plot(x,yhat,label='predicted')
plt.xlabel('x',fontsize=15)
plt.ylabel('$y=x^3+x^2-x-1$',fontsize=15)
plt.legend()

#exercise-1: Replicate the above in Tensorflow 2.x