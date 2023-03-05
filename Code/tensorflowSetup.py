#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:18:00 2023

@author: gouthamgopalakrishna
"""
import tensorflow as tf
print(tf.__version__)
import time
import math


'''
If using google colab, then use following commands
import logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
!pip install tensorflow-gpu==2.6.0
import tensorflow as tf
!pip install keras==2.6.0
'''

#decorators in python
#Given below is an example to calculate time 
def calculate_time(func):
 #the inner 1 function takes arguments through * args and ** kwargs

    def inner1(* args , ** kwargs ):

         # storing time before function execution
         begin = time.time ()
    
         func (* args , ** kwargs )
    
         # storing time after function execution
         end = time.time ()
         print (" Total time taken in : ", func.__name__, end - begin)

    return inner1

# Let 's write a function to compute factorial and wrap it with decorator
@calculate_time
def factorial(num):
    print ( math.factorial(num))

 # calling the function .
factorial (10)


#decorators are used in Tensorflow 2.x to create computational graphs
#Given below is a simple example
def inner_function(x, y, b):
  x = tf. matmul (x, y)
  x = x + b
  return x

# Use the decorator to make `outer _ function ` a `Function `.
@tf . function
def outer_function (x):
  y = tf. constant([[2.0],[3.0]])
  b = tf. constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner _ function ` as well as `outer _ function `.
outer_function(tf.constant([[1.0 , 2.0]])).numpy()


