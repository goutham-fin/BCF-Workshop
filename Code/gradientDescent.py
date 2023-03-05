#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:45 2023

@author: gouthamgopalakrishna
"""
import numpy as np
import matplotlib.pyplot as plt


X = [0.5,2.5] #input
Y = [0.2,0.9] #output
loss = []

def f(x,w,b): #activation function
  return 1/(1+np.exp(-(w*x+b)))

def error(w,b): #loss function
  err = 0.0
  for x,y in zip(X,Y):
    fx = f(x,w,b)
    err += (fx-y)**2
  return 0.5*err

def grad_b(x,w,b,y): #gradient function
  fx = f(x,w,b)
  return (fx-y)*fx*(1-fx)

def grad_w(x,w,b,y): #gradient function
  fx = f(x,w,b)
  return (fx-y)*fx*(1-fx)*x

def do_gradient_descent(): 
  #performs gradient descent
  
  w,b,eta,max_epochs = -4,-4,1.0,3000
  
  for i in range(max_epochs):
    dw,db = 0,0
    for x,y in zip(X,Y):
      dw += grad_w(x,w,b,y)
      db += grad_b(x,w,b,y)
    
    w = w - eta*dw
    b = b - eta*db

  return w,b

##################################################
##################################################
##Momentum gradient descent##
def do_mgd(max_epochs):
    w,b,eta = -2.0,-2.0,0.9
    prev_v_w,prev_v_b,beta = 0,0,0.99
   
    for i in range(max_epochs):
        dw,db = 0,0        
        for x,y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
            
        v_w = beta*prev_v_w+eta*dw
        v_b = beta*prev_v_b+eta*db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
    return w,b

do_mgd(1000)

        
##################################################
##################################################
##Nesterov gradient descent##
loss_nesterov=[]
def do_nag(max_epochs):
    w,b,eta = -2,-2,1.0
    prev_vw,prev_vb,beta = 0,0,0.9
   
    for i in range(max_epochs):
        dw,db = 0,0
        # do partial updates
        v_w = beta*prev_vw
        v_b = beta*prev_vb
        for x,y in zip(X,Y):
            # Look ahead
            dw += grad_w(w-v_w,b-v_b,x,y)
            db += grad_b(w-v_w,b-v_b,x,y)
        vw = beta*prev_vw+eta*dw
        vb = beta*prev_vb+eta*db
        w = w - vw
        b = b - vb
        prev_vw = vw
        prev_vb = vb
        loss_nesterov.append(error(w,b))
    return w,b
do_nag(30000)
plt.plot(loss_nesterov)

##################################################
##################################################
#Exercise-2: Write a function do_adam(max_epochs) that takes max_epochs as inputs
# and computes optimal w and b through adam update rule. Plot the error and compare it 
# with the other methods

        