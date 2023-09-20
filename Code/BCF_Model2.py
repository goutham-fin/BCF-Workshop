# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 00:28:16 2022

@author: gopalakr
"""

import sys
sys.path.insert(0, '../')
from scipy.optimize import fsolve
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 15})
import numpy as np
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import pyplot
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.linewidth'] = 0
import dill
import tensorflow as tf
import time
import glob
import os.path
import argparse 
from scipy.stats import norm
from scipy.stats import t
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
#from keras import backend as K
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


parser = argparse.ArgumentParser(description='Solve the 2D model')
parser.add_argument('-s','--save',type=bool, metavar='',nargs='?',default=True,
                    help='Specify whether to save pickle (warning: requires ~1GB space)')
parser.add_argument('-epochs','--numEpochs',type=int,
                    nargs='?',default=5000,help='Specify number of epochs')
parser.add_argument('-bs','--batchSize',type=int,
                    nargs='?',default=2000,help='Specify batch size')
parser.add_argument('-hidden','--numLayers',type=int,
                    nargs='?',default=5,help='Specify number of layers')
parser.add_argument('-neurons','--numNeurons',type=int,
                    nargs='?',default=30,help='Specify number of neurons per layer')
parser.add_argument('-maxIter','--maxIterations',type=int,
                    nargs='?',default=100,help='Specify number of iterations to run')

args, unknown = parser.parse_known_args()


class nnpde_informed():
    def __init__(self,linearTermE,advection_z_e,advection_f_e,diffusion_z,diffusion_f,cross_term,J0_e,z,f,num_layers,batchSize,lr,adam_iter,dt,agent_type,
                            linearTermH,advection_z_h,advection_f_h,J0_h,params,crisis,X,X_pde):
        self.linearTermE = linearTermE
        self.advection_z_e = advection_z_e
        self.advection_f_e = advection_f_e
        self.diffusion_z = diffusion_z
        self.diffusion_f = diffusion_f
        self.cross_term = cross_term
        self.linearTermH = linearTermH
        self.advection_z_h = advection_z_h
        self.advection_f_h = advection_f_h
        self.J0_e = J0_e
        self.J0_h = J0_h
        self.z,self.f,self.dt = z,f,dt
        self.num_layers = num_layers
        self.lr = lr
        self.adam_iter = adam_iter
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.batchSize = batchSize
        self.number_epochs = adam_iter
        self.lowest_iter= 0
        self.min_loss = 20000
        self.agent_type = agent_type
        self.params=params
        self.crisis = crisis

        self.X,self.X_pde = X,X_pde

    def NN(self,inputDim,num_layers,num_neurons=30):
        model_ = keras.models.Sequential()
        model_.add(keras.layers.Dense(num_neurons,activation='tanh',input_dim=inputDim,kernel_initializer = self.initializer))
        for layer in range(num_layers-1):
            model_.add(keras.layers.Dense(num_neurons,activation='tanh',kernel_initializer=self.initializer))
        model_.add(keras.layers.Dense(1,kernel_initializer=self.initializer))
        return model_

    def get_value_pde(self,value_function_e,value_function_h,X_pde,idx,false_transient,agent_type):
        z,f,t = tf.convert_to_tensor(X_pde[idx,0:1]),tf.convert_to_tensor(X_pde[idx,1:2]),tf.convert_to_tensor(X_pde[idx,2:3])
       
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            tape.watch(f)
            tape.watch(t)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(z)
                tape2.watch(f)
                if agent_type=='households': u= value_function_h(tf.concat([z,f,t],axis=1))
                elif agent_type=='experts': u = value_function_e(tf.concat([z,f,t],axis=1))
            u_z = tape.gradient(u,z,unconnected_gradients=tf.UnconnectedGradients.NONE)
            u_f = tape.gradient(u,f,unconnected_gradients=tf.UnconnectedGradients.NONE)
            u_t = tape.gradient(u,t,unconnected_gradients=tf.UnconnectedGradients.NONE)
        u_zz = tape.gradient(u_z,z,unconnected_gradients=tf.UnconnectedGradients.NONE)
        u_ff = tape.gradient(u_f,f,unconnected_gradients=tf.UnconnectedGradients.NONE)
        u_fz = tape.gradient(u_f,z,unconnected_gradients=tf.UnconnectedGradients.NONE)
        
        if agent_type =='experts':
            advection_z = tf.convert_to_tensor(self.advection_z_e.transpose().reshape(-1,1)[idx])
            advection_f = tf.convert_to_tensor(self.advection_f_e.transpose().reshape(-1,1)[idx])
            linearTerm = tf.convert_to_tensor(self.linearTermE.transpose().reshape(-1,1)[idx])
        elif agent_type=='households':
            advection_z = tf.convert_to_tensor(self.advection_z_h.transpose().reshape(-1,1)[idx])
            advection_f = tf.convert_to_tensor(self.advection_f_h.transpose().reshape(-1,1)[idx])
            linearTerm = tf.convert_to_tensor(self.linearTermH.transpose().reshape(-1,1)[idx])

        diffusion_z = tf.convert_to_tensor(self.diffusion_z.transpose().reshape(-1,1)[idx])
        diffusion_f = tf.convert_to_tensor(self.diffusion_f.transpose().reshape(-1,1)[idx])
        
        cross_term = tf.convert_to_tensor(self.cross_term.transpose().reshape(-1,1)[idx])
        u_pde = u_t + advection_z * u_z + advection_f * u_f + diffusion_z * u_zz + diffusion_f * u_ff + cross_term * u_fz - linearTerm * u
        return u_pde

    def PDESolver(self,value_function_e,value_function_h,j0_e,j0_h,X,X_pde,idx,idx_crisis):
        fpde_e = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'experts')
        
        jE = value_function_e(tf.concat([X[:,0:1],X[:,1:2],X[:,2:3]],axis=1))
        
        
        fpde_h = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'households')
        jH = value_function_h(tf.concat([X[:,0:1],X[:,1:2],X[:,2:3]],axis=1))
        
        #experts
        
        
        loss_1 = tf.reduce_mean(tf.square(tf.reduce_mean(tf.square(fpde_e))))  + tf.reduce_mean(tf.square(tf.nn.relu(-jE))) 
        
        loss_2 = tf.reduce_mean(tf.square(jE - j0_e))

        #households
        loss_3 = tf.reduce_mean(tf.square(jH - j0_h)) 
        loss_4 = tf.reduce_mean(tf.square(fpde_h)) + tf.reduce_mean(tf.square(tf.nn.relu(-jH))) 

        
        loss = loss_1 + loss_2 + loss_3 + loss_4 
        return loss,fpde_e

    def loss_function(self,batchSize):
        idx = np.random.choice(self.X.shape[0],batchSize,replace=True)
        c1,c2 = np.where(self.z==self.z[int(self.crisis[0])])[0][0] - 10,np.where(self.z==self.z[int(self.crisis[-1])])[0][0] + 10
        crisis_points = []
        for i in range(len(self.f)):
            crisis_points = np.hstack((crisis_points,np.linspace(c1+i*len(self.z),c2+i*len(self.z),len(self.crisis))))
        idx_crisis = np.random.choice(len(crisis_points),100,replace=True)
        loss_total,fpde = self.PDESolver(self.value_function_e,self.value_function_h,self.J0_e,self.J0_h,self.X,self.X_pde,idx,idx_crisis)
        return loss_total,fpde

    @tf.function 
    def training_step(self):
        with tf.GradientTape(persistent=True) as tape:
            loss_total,fpde= self.loss_function(self.batchSize)
        grads_value_e = tape.gradient(loss_total,self.theta_valueFunction_e)
        grads_value_h = tape.gradient(loss_total,self.theta_valueFunction_h)
        self.optimizer.apply_gradients(zip(grads_value_e,self.theta_valueFunction_e))
        self.optimizer.apply_gradients(zip(grads_value_h,self.theta_valueFunction_h))
        return loss_total,fpde

    def train(self):
        if os.path.isfile('./save/value_function_experts.h5'):
            self.value_function_e = load_model('./save/value_function_experts.h5')
        else:
            self.value_function_e = self.NN(3,self.params['num_layers'])
        
        if os.path.isfile('./save/value_function_households.h5'):
            self.value_function_h = load_model('./save/value_function_households.h5')
        else:
            self.value_function_h = self.NN(3,self.params['num_layers'])

        self.theta_valueFunction_e = self.value_function_e.trainable_variables
        self.theta_valueFunction_h = self.value_function_h.trainable_variables
        
        self.best_valueFunction_e = tf.keras.models.clone_model(self.value_function_e)
        self.best_valueFunction_h = tf.keras.models.clone_model(self.value_function_h)

        self.LVF = []

        min_loss = float('inf')
        
        start_time = time.time()
        for epoch in range(self.number_epochs+1):
                loss_total,fpde = self.training_step()
                if (loss_total<min_loss):
                #if True:
                    self.lowest_iter=epoch
                    min_loss = loss_total
                    self.best_valueFunction_e.set_weights(self.value_function_e.get_weights())
                    self.best_valueFunction_h.set_weights(self.value_function_h.get_weights())
                    elapsed_time = time.time() - start_time
                if epoch%1000==0: print('It: %d, Loss: %.3e, Time: %.2f'% 
                          (epoch, min_loss.numpy(), elapsed_time))
                self.LVF.append(loss_total.numpy())
        self.value_function_e.set_weights(self.best_valueFunction_e.get_weights())
        self.value_function_h.set_weights(self.best_valueFunction_h.get_weights())
        self.fpde = fpde
    
    def predict(self,x_star):
        je_new = self.value_function_e(tf.concat([x_star[:,0:1],x_star[:,1:2],x_star[:,2:3]],axis=1))
        jh_new = self.value_function_h(tf.concat([x_star[:,0:1],x_star[:,1:2],x_star[:,2:3]],axis=1))
        return je_new.numpy(),jh_new.numpy()


class model_nnpde():
    def __init__(self,params):
        self.params = params
        self.Nz = 1000 #z denotes wealth-share (endogenous state variable)
        self.z = np.linspace(0.001,0.999, self.Nz)
        self.Nf = 30 #f denotes productivity of experts (exogenous state variable)
        self.f = np.linspace(self.params['f_l'], self.params['f_u'], self.Nf)
        
        self.dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        self.dz2 = self.dz[0:self.Nz-2]**2;
        self.z_mat = np.tile(self.z,(self.Nf,1)).transpose()
        self.dz_mat = np.tile(self.dz,(self.Nf,1)).transpose()
        self.dz2_mat = np.tile(self.dz2,(self.Nf,1)).transpose()
        self.df = self.f[1:self.Nf] - self.f[0:self.Nf-1]
        self.df2 = self.df[0:self.Nf-2]**2
        self.f_mat = np.tile(self.f,(self.Nz,1))
        self.df_mat = np.tile(self.df,(self.Nz,1))
        self.df2_mat = np.tile(self.df2,(self.Nz,1))

        self.q   =  np.array(np.tile(1,(self.Nz,self.Nf)),dtype=np.float64); 
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qf = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.qff = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.Qfz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        self.thetah=  np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.theta= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.r = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.ssq= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.ssf= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.chi= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.iota= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.Je = np.ones([self.Nz,self.Nf]) 
        self.Jh = np.ones([self.Nz,self.Nf]) 
        self.exit_flag = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
        self.Jtilde_z= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.Jtilde_f= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);
        self.tau= np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64);

        self.first_time = np.linspace(0,0,self.Nf)
        self.psi = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64) 
        
        self.maxIterations=150
        
        if self.params['scale']>1: self.convergenceCriterion = 1e-4;
        else: self.convergenceCriterion = 1e-4 ;
        self.converged = False
        self.Iter=0
        try:
            if not os.path.exists('../../output'):
                os.mkdir('../../output')
        except:
            print('Warning: Cannot create directory for plots')
        self.amax = np.float('Inf')
        self.amax_vec=[]
        
        z_tile = np.tile(self.z,len(self.f))
        f_tile = np.repeat(self.f,len(self.z)) 
        self.dt = 0.9
        self.X = np.vstack((z_tile,f_tile,np.full(z_tile.shape[0],self.dt))).transpose().astype(np.float64)
        self.X_pde = np.vstack((z_tile,f_tile,np.random.uniform(0,self.dt,z_tile.shape[0]))).transpose().astype(np.float64)
        self.x_star = np.vstack((z_tile,f_tile,np.full(z_tile.shape[0],0))).transpose()
                
        
    def equations_region1(self,q_p, Psi_p, sig_qk_p, sig_qf_p, zi, fi):
        i_p = (q_p - 1)/self.params['kappa']
        eq1 = (self.f[fi]-self.params['aH'])/q_p -\
                self.params['alpha'] * self.Jtilde_z[zi,fi]*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi])*(sig_qk_p**2 + sig_qf_p**2 + 2*self.params['corr']*sig_qk_p*sig_qf_p) - self.params['alpha']* self.Jtilde_f[zi,fi]*self.sig_f[zi,fi]*(sig_qf_p + self.params['corr']*sig_qk_p)
        eq2 = (self.params['rho']*self.z_mat[zi,fi] + self.params['rho']*(1-self.z_mat[zi,fi])) * q_p  - Psi_p * (self.f[fi] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
              
        eq3 = sig_qk_p - sig_qk_p*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.params['sigma']

        if fi==0:
            eq4 = sig_qf_p * self.q[zi-1,fi]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1]  + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])
        else:
            eq4 = sig_qf_p * self.q[zi-1,fi]/(q_p * self.dz[zi-1])*(self.params['alpha'] * Psi_p - self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1] + self.sig_f[zi,fi] * self.q[zi,fi-1]/(q_p * self.df[fi-1]) + sig_qf_p - sig_qf_p/self.dz[zi-1]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])

        ER = np.array([eq1,eq2,eq3,eq4])
        QN = np.zeros(shape=(4,4))

        QN[0,:] = np.array([-self.params['alpha']**2 * self.Jtilde_z[zi,fi]*(sig_qk_p**2 + sig_qf_p**2 + sig_qk_p*sig_qf_p*self.params['corr']*2), -2*self.params['alpha']*self.Jtilde_z[zi,fi]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi])*sig_qk_p - 2*self.params['alpha']*self.Jtilde_z[zi,fi]*self.params['corr']*(self.params['alpha']*Psi_p - self.z_mat[zi,fi])*sig_qf_p -self.params['alpha']*self.Jtilde_f[zi,fi]*self.params['corr']*self.sig_f[zi,fi], \
                            -2* self.params['alpha'] * self.Jtilde_z[zi,fi]*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])*sig_qf_p - 2*self.params['corr']*self.params['alpha']*sig_qk_p*self.Jtilde_z[zi,fi]*(self.params['alpha']*Psi_p - self.z_mat[zi,fi]) - self.params['alpha']*self.Jtilde_f[zi,fi]*self.sig_f[zi,fi], -(self.f[fi]-self.params['aH'])/(q_p**2)])
        QN[1,:] = np.array([self.params['aH'] - self.f[fi], 0, 0,  self.params['rho'] * self.z_mat[zi,fi] + (1-self.z_mat[zi,fi])*self.params['rho'] + 1/self.params['kappa']])
        QN[2,:] = np.array([-sig_qk_p * self.params['alpha']/self.dz[zi-1]*(1-self.q[zi-1,fi]/q_p), 1-((self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1])*(q_p - self.q[zi-1,fi])/q_p, \
                                 0, -sig_qk_p*(self.q[zi-1,fi]/q_p**2)*(self.params['alpha'] * Psi_p-self.z_mat[zi,fi])/self.dz[zi-1]])
        if fi==0:
            QN[3,:] = np.array([-sig_qf_p/self.dz[zi-1] + sig_qf_p/self.dz[zi-1] * self.q[zi-1,fi]/q_p, 0, 1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) ,-sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,fi])])
        else:
            QN[3,:] = np.array([-sig_qf_p/self.dz[zi-1] + sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/q_p, 0, \
                                1-1/self.dz[zi-1]*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(self.params['alpha']* Psi_p-self.z_mat[zi,fi]) - self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p**2 * self.df[fi-1]) ])
        
        EN = np.array([Psi_p, sig_qk_p, sig_qf_p, q_p]) - np.linalg.solve(QN,ER)
        del ER, QN
        return EN
    
    def equations_region2(self,q_p,sig_qk_p,sig_qf_p,Chi_p_old,zi,fi):
        error = 100
        iter = 1
        while error>0.001 or iter<10: 
            i_p = (q_p-1)/self.params['kappa']
            eq1 = self.params['rho'] * q_p  - (self.f[fi] - i_p)
            eq2 = sig_qk_p - sig_qk_p*(Chi_p_old-self.z_mat[zi,fi])/self.dz[zi-1] + (sig_qk_p)*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(Chi_p_old - self.z_mat[zi,fi]) - self.params['sigma']

            if fi==0:
                eq3 = sig_qf_p*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(Chi_p_old-self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1]  + sig_qf_p - sig_qf_p/self.dz[zi-1]*(Chi_p_old-self.z_mat[zi,fi])
            else:
                eq3 = sig_qf_p*self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(Chi_p_old-self.z_mat[zi,fi]) - self.sig_f[zi,fi]/self.df[fi-1] + self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p*self.df[fi-1]) + sig_qf_p - sig_qf_p/self.dz[zi-1]*(Chi_p_old-self.z_mat[zi,fi])
            ER = np.array([eq1,eq2,eq3])
            QN = np.zeros(shape=(3,3))
            QN[0,:] = np.array([0,0,self.params['rho']*self.z_mat[zi,fi] + (Chi_p_old-self.z_mat[zi,fi])*self.params['rho'] + 1/self.params['kappa']])
            QN[1,:] = np.array([1-(Chi_p_old-self.z_mat[zi,fi])/self.dz[zi-1] + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(Chi_p_old-self.z_mat[zi,fi]), 0, -sig_qk_p*(self.q[zi-1,fi]/q_p**2)*(Chi_p_old-self.z_mat[zi,fi])/self.dz[zi-1]])
            if fi==0:
                QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(1-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(Chi_p_old-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(Chi_p_old-self.z_mat[zi,fi])])
            else:
                QN[2,:] = np.array([0,1-1/self.dz[zi-1]*(Chi_p_old-self.z_mat[zi,fi]) + self.q[zi-1,fi]/(q_p*self.dz[zi-1])*(1-self.z_mat[zi,fi]), -sig_qf_p/self.dz[zi-1]*self.q[zi-1,fi]/(q_p**2)*(Chi_p_old-self.z_mat[zi,fi]) - self.sig_f[zi,fi]*self.q[zi,fi-1]/(q_p**2 * self.df[fi-1])])
          
            EN = np.array([sig_qk_p,sig_qf_p,q_p]) - np.linalg.solve(QN,ER)
            sig_qk_p,sig_qf_p,q_p = EN[0], EN[1], EN[2]
            omega_1 = sig_qk_p**2 + sig_qf_p**2 + 2*self.params['corr']*sig_qk_p*sig_qf_p
            omega_2 = sig_qk_p*self.params['corr'] + sig_qf_p
            Chi_p = self.z_mat[zi,fi] - (self.Jtilde_f[zi,fi]*self.sig_f[zi,fi]*omega_2)/(self.Jtilde_z[zi,fi]*omega_1)
            error = np.abs(Chi_p - Chi_p_old)
            if Chi_p < self.params['alpha']:
                Chi_p = self.params['alpha']
                break
            else: Chi_p_old = Chi_p.copy()
            #print(error,Chi_p_old,Chi_p)
            del ER,QN
            iter+=1
        if Chi_p <= self.params['alpha']: Chi_p = self.params['alpha']
        return sig_qk_p,sig_qf_p,q_p,Chi_p
    
    def pickle_stuff(self,object_name,filename):
                    with open(filename,'wb') as f:
                        dill.dump(object_name,f)
    
    def solve(self,pde='True'):
        self.psi[0,:]=0
        self.q[0,:] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[0,:]*(self.f-self.params['aH'])))/(1 + self.params['kappa']*(self.params['rho'] + self.z[0] * (self.params['rho'] - self.params['rho'])));
        self.chi[0,:] = self.params['alpha'];
        self.ssq[0,:] = self.params['sigma'];
        self.ssf[0,:] = 0
        self.q0 = (1 + self.params['kappa'] * self.params['aH'])/(1 + self.params['kappa'] * self.params['rho']); 
        self.iota[0,:] = (self.q0-1)/self.params['kappa']
        self.sig_f = self.params['beta_f'] * (self.params['f_u'] - self.f_mat)*(self.f_mat-self.params['f_l'])

        for timeStep in range(self.maxIterations):
            self.Iter+=1            
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe_z = np.vstack([((self.logValueE[1,:]-self.logValueE[0,:])/(self.z_mat[1,:]-self.z_mat[0,:])).reshape(-1,self.Nf),(self.logValueE[2:,:]-self.logValueE[0:-2,:])/(self.z_mat[2:,:]-self.z_mat[0:-2,:]),((self.logValueE[-1,:]-self.logValueE[-2,:])/(self.z_mat[-1,:]-self.z_mat[-2,:])).reshape(-1,self.Nf)]);
            self.dLogJh_z = np.vstack([((self.logValueH[1,:]-self.logValueH[0,:])/(self.z_mat[1,:]-self.z_mat[0,:])).reshape(-1,self.Nf),(self.logValueH[2:,:]-self.logValueH[0:-2,:])/(self.z_mat[2:,:]-self.z_mat[0:-2,:]),((self.logValueH[-1,:]-self.logValueH[-2,:])/(self.z_mat[-1,:]-self.z_mat[-2,:])).reshape(-1,self.Nf)]);
            self.dLogJe_f = np.hstack([((self.logValueE[:,1]-self.logValueE[:,0])/(self.f_mat[:,1]-self.f_mat[:,0])).reshape(self.Nz,-1),(self.logValueE[:,2:]-self.logValueE[:,0:-2])/(self.f_mat[:,2:]-self.f_mat[:,0:-2]),((self.logValueE[:,-1]-self.logValueE[:,-2])/(self.f_mat[:,-1]-self.f_mat[:,-2])).reshape(self.Nz,1)]);
            self.dLogJh_f = np.hstack([((self.logValueH[:,1]-self.logValueH[:,0])/(self.f_mat[:,1]-self.f_mat[:,0])).reshape(self.Nz,1),(self.logValueH[:,2:]-self.logValueH[:,0:-2])/(self.f_mat[:,2:]-self.f_mat[:,0:-2]),((self.logValueH[:,-1]-self.logValueH[:,-2])/(self.f_mat[:,-1]-self.f_mat[:,-2])).reshape(self.Nz,1)]);
            if self.params['scale']>1:
                self.Jtilde_z = (1-self.params['gamma'])*self.dLogJh_z - (1-self.params['gamma'])*self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_f = (1-self.params['gamma'])*self.dLogJh_f - (1-self.params['gamma'])*self.dLogJe_f
            else:
                self.Jtilde_z = self.dLogJh_z - self.dLogJe_z + 1/(self.z_mat*(1-self.z_mat))
                self.Jtilde_f = self.dLogJh_f - self.dLogJe_f
            self.crisis = np.zeros(self.f.shape)
            for fi in range(self.Nf):
                for zi in range(1,self.Nz):
                    
                    if self.psi[zi-1,fi]<1:
                        result= self.equations_region1(self.q[zi-1,fi], self.psi[zi-1,fi], self.ssq[zi-1,fi], self.ssf[zi-1,fi], zi, fi)
                        if result[0]>=1:
                            #break #for debugging purpose
                            self.crisis[fi]=zi
                            self.psi[zi,fi]=1
                            self.chi[zi,fi] = self.params['alpha']
                            result = self.equations_region2(self.q[zi-1,fi],self.ssq[zi-1,fi],self.ssf[zi-1,fi],self.chi[zi-1,fi],zi,fi)
                            self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi], self.chi[zi,fi] = result[0], result[1], result[2], result[3]
                            del result
                        else:
                            self.psi[zi,fi], self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi] =result[0], result[1], result[2], result[3]
                            self.chi[zi,fi] = self.params['alpha']
                            del(result)
                    else:
                        self.psi[zi,fi]=1
                        result = self.equations_region2(self.q[zi-1,fi],self.ssq[zi-1,fi],self.ssf[zi-1,fi],self.chi[zi-1,fi],zi,fi)
                        self.ssq[zi,fi], self.ssf[zi,fi], self.q[zi,fi],self.chi[zi,fi] = result[0], result[1], result[2],result[3]
                        del result
            #fix numerical error
            self.ssf[1,:] = self.ssf[2,:]
            self.ssq[1:3,:] = self.ssq[0,:]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
            self.crisis_flag_bound = np.array(np.tile(0,(self.Nz,self.Nf)), dtype = np.float64)
            
            for j in range(self.Nf): 
                self.crisis_flag[0:int(self.crisis[j]),j] = 1
                
            def last_nonzero(arr, axis, invalid_val=-1):
                '''
                not used for still useful for other purposes
                '''
                mask = arr!=0
                val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
                return np.where(mask.any(axis=axis), val, invalid_val)
            
            
            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qf[:,1:self.Nf]  = (self.q [:,1:self.Nf] - self.q [:,0:self.Nf-1])/self.df_mat; self.qf[:,0]=self.qf[:,1];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2_mat); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qff[:,2:self.Nf] = (self.q[:,2:self.Nf] + self.q[:,0:self.Nf-2] - 2.*self.q[:,1:self.Nf-1])/(self.df2_mat); self.qff[:,0]=self.qff[:,2]; self.qff[:,1]=self.qff[:,2]
            q_temp = np.row_stack((self.q[0,:],self.q,self.q[self.q.shape[0]-1,:]))
            q_temp = np.column_stack((q_temp[:,0],q_temp,q_temp[:,q_temp.shape[1]-1]))
            for fi in range(1,self.Nf):
                for zi in range(1,self.Nz):
                            self.Qfz[zi,fi]= (q_temp[zi+1,fi+1] - q_temp[zi+1,fi-1] - q_temp[zi-1,fi+1] + q_temp[zi-1,fi-1])/(4*self.df_mat[zi-1,fi-1]*self.dz_mat[zi-1,fi-1]);
            del(q_temp)
            self.qzl  = self.qz/self.q; 
            self.qfl  = self.qf /self.q; 
            self.qzzl = self.qzz/ self.q;
            self.qffl = self.qff/self.q;
            self.qfzl = self.Qfz/self.q;
            
            self.iota = (self.q-1)/self.params['kappa']
            self.theta = self.chi*self.psi/self.z_mat
            self.thetah = (1-self.chi*self.psi)/(1-self.z_mat)
            self.theta[0] = self.theta[1]
            self.thetah[0] = self.thetah[1]
            
            
            self.consWealthRatioE = self.params['rho']
            self.consWealthRatioH = self.params['rho']
            self.sig_zk = self.z_mat*(self.theta-1)*self.ssq
            self.sig_zf = self.z_mat*(self.theta-1)*self.ssf
            self.sig_jk_e = self.dLogJe_z*self.sig_zk
            self.sig_jf_e = self.dLogJe_f*self.sig_f + self.dLogJe_z*self.sig_zf
            self.sig_jk_h = self.dLogJh_z*self.sig_zk
            self.sig_jf_h = self.dLogJh_f*self.sig_f + self.dLogJh_z*self.sig_zf
            self.priceOfRiskE_k = -(1-self.params['gamma'])*self.sig_jk_e + self.sig_zk/self.z_mat + self.ssq + (self.params['gamma']-1)*self.params['sigma']
            self.priceOfRiskE_f = -(1-self.params['gamma'])*self.sig_jf_e + self.sig_zf/self.z_mat + self.ssf
            self.priceOfRiskH_k = -(1-self.params['gamma'])*self.sig_jk_h - 1/(1-self.z_mat)*self.sig_zk + self.ssq + self.params['gamma']*self.params['sigma']
            self.priceOfRiskH_f = -(1-self.params['gamma'])*self.sig_jf_h - 1/(1-self.z_mat)*self.sig_zf + self.ssf
            self.priceOfRiskE_hat1 = self.priceOfRiskE_k + self.params['corr']*self.priceOfRiskE_f
            self.priceOfRiskE_hat2 = self.params['corr']* self.priceOfRiskE_k + self.priceOfRiskE_f
            self.priceOfRiskH_hat1 = self.priceOfRiskH_k + self.params['corr']*self.priceOfRiskH_f
            self.priceOfRiskH_hat2 = self.params['corr']* self.priceOfRiskH_k + self.priceOfRiskH_f
            
            self.rp = self.ssq*self.priceOfRiskE_hat1 + self.ssf*self.priceOfRiskE_hat2
            self.rp_ = self.ssq*self.priceOfRiskH_hat1 + self.ssf*self.priceOfRiskH_hat2
            self.rp_1 = self.params['alpha']*self.rp + (1-self.params['alpha'])*self.rp_
            
            
            self.mu_z = self.z_mat*( (self.f_mat - self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*(self.ssq*(self.priceOfRiskE_hat1 - self.ssq) + self.ssf*(self.priceOfRiskE_hat2 - self.ssf) - 2* self.params['corr'] * self.ssq*self.ssf ) + (1-self.params['alpha'])*(self.ssq*(self.priceOfRiskE_hat1 - self.priceOfRiskH_hat1) + self.ssf*(self.priceOfRiskE_hat2 - self.priceOfRiskH_hat2))) +\
                            self.params['lambda_d']*(self.params['zbar']-self.z_mat) 
            for fi in range(self.Nf):
                crisis_temp = np.where(self.crisis_flag[:,fi]==1.0)[0][-1]+1
                try:
                    self.mu_z[crisis_temp,fi] = self.mu_z[crisis_temp-1,fi]
                except:
                    print('no crisis')
            
            self.mu_f = self.params['pi']*(self.params['f_avg'] - self.f_mat)
            self.growthRate = np.log(self.q)/self.params['kappa'] -self.params['delta']
            self.sig_zk[0]=0 #latest change
            self.ssTotal = self.ssf + self.ssq
            self.sig_zTotal = self.sig_zk + self.sig_zf
            self.priceOfRiskETotal = self.priceOfRiskE_k + self.priceOfRiskE_f
            self.priceOfRiskHTotal = self.priceOfRiskH_k + self.priceOfRiskH_f
            self.Phi = np.log(self.q)/self.params['kappa']
            self.mu_q = self.qzl*self.mu_z + self.qfl*self.mu_f + 0.5*self.qzzl*(self.sig_zk**2 + self.sig_zf**2 + 2*self.params['corr']*self.sig_zk*self.sig_zf) +\
                    0.5*self.qffl*self.sig_f**2 + self.qfzl*(self.sig_zk*self.sig_f*self.params['corr'] + self.sig_zf * self.sig_f)
            self.r = self.crisis_flag*(-self.rp_ + (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma']*(self.ssq-self.params['sigma']) + self.params['corr'] * self.params['sigma'] * self.ssf) +\
                    (1-self.crisis_flag)*(-self.rp + (self.f_mat - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq-self.params['sigma']) + self.params['corr'] * self.params['sigma'] * self.ssf)
            
            for fi in range(self.Nf):
                crisis_temp = np.where(self.crisis_flag[:,fi]==1.0)[0][-1]+1
                try:
                    self.r[crisis_temp-1:crisis_temp+2,fi] = 0.5*(self.r[crisis_temp+3,fi] + self.r[crisis_temp-2,fi]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
                except:
                    print('no crisis')
            self.A = self.psi*(self.f_mat) + (1-self.psi) * (self.params['aH'])
            self.AminusIota = self.psi*(self.f_mat - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.rp_2 = self.AminusIota/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma']*(self.ssq-self.params['sigma'])+self.params['corr']*self.params['sigma']*self.ssf - self.r
            self.pd = np.log(self.q / self.AminusIota)
            self.vol = np.sqrt(self.ssq**2 + self.ssf**2)
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.f_mat - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.Jhat_e = self.Je.copy().reshape(self.Nz,self.Nf)
            self.Jhat_h = self.Jh.copy().reshape(self.Nz,self.Nf)
            self.diffusion_z = 0.5*(self.sig_zk**2 + self.sig_zf**2 + 2*self.params['corr']*self.sig_zk*self.sig_zf)
            self.diffusion_f = 0.5*(self.sig_f)**2
            if self.params['scale']>1:
                self.advection_z_e = self.mu_z 
                self.advection_f_e = self.mu_f 
                self.advection_z_h = self.mu_z
                self.advection_f_h = self.mu_f 
                self.linearTermE = -(0.5*self.params['gamma']*(self.sig_jk_e**2 + self.sig_jf_e**2 + 2*self.params['corr']*self.sig_jk_e*self.sig_jf_e + self.params['sigma']**2) -\
                                    self.growthRate + (self.params['gamma']-1)*(self.sig_jk_e*self.params['sigma'] + self.params['corr']*self.params['sigma']*self.sig_jf_e) -\
                                    self.params['rho']*(np.log(self.params['rho']) - np.log(self.Jhat_e) + np.log(self.z_mat*self.q))) 
                self.linearTermH = -(0.5*self.params['gamma']*(self.sig_jk_h**2 + self.sig_jf_h**2 + 2*self.params['corr']*self.sig_jk_h*self.sig_jf_h + self.params['sigma']**2) -\
                                    self.growthRate + (self.params['gamma']-1)*(self.sig_jk_h*self.params['sigma'] + self.params['corr']*self.params['sigma']*self.sig_jf_h) -\
                                    self.params['rho']*(np.log(self.params['rho']) - np.log(self.Jhat_h) + np.log((1-self.z_mat)*self.q)))
            else:
                self.advection_z_e = self.mu_z + (1-self.params['gamma'])*(self.params['sigma']*self.sig_zk + self.params['sigma']*self.sig_zf)
                self.advection_f_e = self.mu_f + (1-self.params['gamma'])*self.params['corr']*self.params['sigma']*self.sig_f
                self.advection_z_h = self.mu_z + (1-self.params['gamma'])*(self.params['sigma']*self.sig_zk + self.params['sigma']*self.sig_zf)
                self.advection_f_h = self.mu_f + (1-self.params['gamma'])*self.params['corr']*self.params['sigma']*self.sig_f            
                self.linearTermE = (1-self.params['gamma']) * (self.growthRate - 0.5*self.params['gamma']*self.params['sigma']**2 +\
                                  self.params['rho']*(np.log(self.params['rho']) + np.log(self.q*self.z_mat))) -  self.params['rho']* np.log(self.Je)
                self.linearTermH = (1-self.params['gamma']) * (self.growthRate - 0.5*self.params['gamma']*self.params['sigma']**2  +\
                                  self.params['rho']*(np.log(self.params['rho']) + np.log(self.q*(1-self.z_mat)))) -   self.params['rho'] * np.log(self.Jh)

            self.cross_term = self.sig_zk*self.sig_f*self.params['corr'] + self.sig_zf*self.sig_f
            
            #Time step
            #data prep
            if pde=='True':
                learning_rate = 0.001
                layers = self.params['num_layers']
                self.dt = 0.9
                Jhat_e0 = self.Jhat_e.transpose().flatten().reshape(-1,1)
                Jhat_h0 = self.Jhat_h.transpose().flatten().reshape(-1,1)
                self.crisis_z = np.linspace(ext.z[int(ext.crisis[0])],ext.z[int(ext.crisis[-1])],50)
                
                #sovle the PDE
                print('Solving for value functions')
                model = nnpde_informed(-self.linearTermE.transpose().reshape(-1,1), self.advection_z_e.transpose().reshape(-1,1),self.advection_f_e.transpose().reshape(-1,1), 
                                         self.diffusion_z.transpose().reshape(-1,1),self.diffusion_f.transpose().reshape(-1,1),self.cross_term.transpose().reshape(-1,1), 
                                         Jhat_e0.reshape(-1,1).astype(np.float64),self.z,self.f,layers,self.params['batchSize'],learning_rate,int(self.params['epochs']/(np.sqrt(timeStep+1))),self.dt,'households',
                                         -self.linearTermH.transpose().reshape(-1,1), self.advection_z_h.transpose().reshape(-1,1),self.advection_f_h.transpose().reshape(-1,1),Jhat_h0.reshape(-1,1).astype(np.float64),self.params,self.crisis,self.X,self.X_pde)
                model.train()
                newJeraw,newJhraw = model.predict(self.x_star)
                model.value_function_e.save('./save/value_function_experts.h5')
                model.value_function_h.save('./save/value_function_households.h5')
                self.fpde_e = model.fpde
                del model 
                newJe = newJeraw.transpose().reshape(self.Nf,self.Nz).transpose()
                newJh = newJhraw.transpose().reshape(self.Nf,self.Nz).transpose()
               

                #upate Je and Jh for next static step iteration
                
                self.ChangeJe = np.abs(newJe - self.Je)
                self.ChangeJh = np.abs(newJh - self.Jh)
                
                cutoff=1
                self.relChangeJe = np.abs((newJe[cutoff:-cutoff,:] - self.Je[cutoff:-cutoff,:]) / self.Je[cutoff:-cutoff,:])
                self.relChangeJh = np.abs((newJh[cutoff:-cutoff,:] - self.Jh[cutoff:-cutoff,:]) / self.Jh[cutoff:-cutoff,:])
                #break if nan values
                if np.sum(np.isnan(newJe))>0 or np.sum(np.isnan(newJh))>0:
                    print('NaN values found in Value function')
                    break
                self.Jh = np.maximum(0.001,newJh)
                self.Je = newJe
                
                if self.params['write_pickle']==True:
                    self.pickle_stuff(self,'model2D' + '.pkl') 
                
               
                self.max = np.maximum(np.mean(self.ChangeJe),np.mean(self.ChangeJh))
                self.relmax = np.maximum(np.mean(self.relChangeJe),np.mean(self.relChangeJh))
                self.amax = np.minimum(self.max,self.relmax)
                if self.amax < self.convergenceCriterion:
                    self.converged = 'True'
                    break
                elif len(self.amax_vec)>1 and np.abs(self.amax - self.amax_vec[-1])>0.5:
                    print('check inner loop. error is very large: ',self.amax)
                    break
                print('Iteration number and mean error: ',self.Iter,',',self.amax)
                print('/n')
                self.amax_vec.append(self.amax)
                

    def clear_saved_models(self):            
                files = glob.glob('./save/*')
                for f in files:
                    os.remove(f)
                
            
    def plots_(self):
        try:
            if not os.path.exists('../output/extended'):
                os.mkdir('../output/extended')
        except:
            print('Warning: Cannot create directory for plots')
            return
        plot_path = '../output/plots/extended/'
        index1 = np.where(self.f==min(self.f, key=lambda x:abs(x-self.params['f_l'])))[0][0]
        index2=  np.where(self.f==min(self.f, key=lambda x:abs(x-(self.params['f_l']+self.params['f_u'])/2)))[0][0]
        index3 = np.where(self.f==min(self.f, key=lambda x:abs(x-self.params['f_u'])))[0][0]
        
        vars = ['self.q','self.theta','self.thetah','self.psi','self.ssq','self.ssf','self.mu_z','self.sig_zk','self.sig_zf','self.priceOfRiskE_k','self.priceOfRiskE_f','self.priceOfRiskH_k','self.priceOfRiskH_f','self.rp','self.vol']
        labels = ['q','$\theta_{e}$','$\theta_{h}$','$\psi$','$\sigma + \sigma^{q,k}$','\sigma^{q,a}', '$\mu^z$','$\sigma^{z,k}$','$\sigma^{z,f}$','$\zeta_{e}^k$', '$\zeta_{e}^f$','$\zeta_{h}^k$','$\zeta_{h}^f$','$\mu_e^R -r$','$\norm{\sigma^R}$']
        title = ['Price','Portfolio Choice: Experts', 'Portfolio Choice: Households',\
                     'Capital Share: Experts', 'Price return diffusion (capital shock)','Price return diffusion (productivity shock)','Drift of wealth share: Experts',\
                     'Diffusion of wealth share (capital shock)','Diffusion of wealth share (productivity shock)', 'Experts price of risk: capital shock','Experts price of risk: productivity shock',\
                     'Household price of risk: capital shock','Household price of risk: productivity shock','Risk premium']
        
        for i in range(len(vars)):
            plt.plot(self.z[1:],eval(vars[i])[1:,index1],label=r'$a_e$={i}'.format(i=str(round(self.f[int(index1)],2))))
            plt.plot(self.z[1:],eval(vars[i])[1:,int(index2)],label='$a_e$={i}'.format(i= str(round(self.f[int(index2)]))))
            plt.plot(self.z[1:],eval(vars[i])[1:,int(index3)],label='$a_e$={i}'.format(i= str(round(self.f[int(index3)],2))),color='b') 
            plt.grid(True)
            plt.legend(loc=0)
            #plt.axis('tight')
            plt.xlabel('Wealth share (z)')
            plt.ylabel(labels[i])
            plt.title(title[i],fontsize = 20)
            plt.rc('legend', fontsize=15) 
            plt.rc('axes',labelsize = 15)
            plt.savefig(plot_path + str(vars[i]).replace('self.','') + '_extended.png',dpi=100,figsize=(10, 7))
            plt.figure()
            
    
            
if __name__ =="__main__":
    params = {'rho': 0.05, 'aH': 0.03,
            'alpha':0.65, 'kappa':5, 'delta':0.05, 'zbar':0.1,
            'lambda_d':0.03, 'sigma':0.1, 'gamma':5,  'corr':0.5,
             'pi' : 0.01, 'f_u' : 0.2, 'f_l' : 0.1,
             'scale':2,'batchSize':1000,
            'DGM':False,'num_neurons':256,'num_layers':4}
    params['beta_f'] = 0.25/params['sigma']
    params['f_avg'] = (params['f_l'] + params['f_u'])/2
    params['write_pickle'] = True
    params['batchSize'] = args.batchSize
    params['epochs'] = 6000
    ext = model_nnpde(params)
    ext.maxIterations = args.maxIterations
    ext.solve(pde='True')
    ext.clear_saved_models()
    
    
    
    
    
