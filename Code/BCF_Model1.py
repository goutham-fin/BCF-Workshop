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
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='Solve the 1D model')
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
                    nargs='?',default=70,help='Specify number of iterations to run')

args, unknown = parser.parse_known_args()


class nnpde_informed_B1():
    def __init__(self,linearTermE,advection_z_e,diffusion_z,J0_e,z,num_layers,batchSize,lr,adam_iter,dt,
                            linearTermH,advection_z_h,J0_h,params,X,X_pde):
        self.linearTermE = linearTermE
        self.advection_z_e = advection_z_e
        self.diffusion_z = diffusion_z
        self.linearTermH = linearTermH
        self.advection_z_h = advection_z_h
        self.J0_e = J0_e
        self.J0_h = J0_h
        self.z,self.dt = z,dt
        self.num_layers = num_layers
        self.lr = lr
        self.adam_iter = adam_iter
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.batchSize = batchSize
        self.number_epochs = adam_iter
        self.lowest_iter= 0
        self.min_loss = 20000
        self.params=params

        
        self.X,self.X_pde = X,X_pde

    def NN(self,inputDim,num_layers,num_neurons=30):
        model_ = keras.models.Sequential()
        model_.add(keras.layers.Dense(num_neurons,activation='tanh',input_dim=inputDim,kernel_initializer = self.initializer))
        for layer in range(num_layers-1):
            model_.add(keras.layers.Dense(num_neurons,activation='tanh',kernel_initializer=self.initializer))
        model_.add(keras.layers.Dense(1,kernel_initializer=self.initializer))
        return model_
    
    def get_value_pde(self,value_function_e,value_function_h,X_pde,idx,false_transient,agent_type):
        z,t = tf.convert_to_tensor(X_pde[idx,0:1]),tf.convert_to_tensor(X_pde[idx,1:2])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z)
            tape.watch(t)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(z)
                if agent_type=='households': u= value_function_h(tf.concat([z,t],axis=1))
                elif agent_type=='experts': u = value_function_e(tf.concat([z,t],axis=1))
            u_z = tape.gradient(u,z,unconnected_gradients=tf.UnconnectedGradients.NONE)
            u_t = tape.gradient(u,t,unconnected_gradients=tf.UnconnectedGradients.NONE)
        u_zz = tape.gradient(u_z,z,unconnected_gradients=tf.UnconnectedGradients.NONE)
        
        if agent_type =='experts':
            advection_z = tf.convert_to_tensor(self.advection_z_e.transpose().reshape(-1,1)[idx])
            linearTerm = tf.convert_to_tensor(self.linearTermE.transpose().reshape(-1,1)[idx])
        elif agent_type=='households':
            advection_z = tf.convert_to_tensor(self.advection_z_h.transpose().reshape(-1,1)[idx])
            linearTerm = tf.convert_to_tensor(self.linearTermH.transpose().reshape(-1,1)[idx])

        diffusion_z = tf.convert_to_tensor(self.diffusion_z.transpose().reshape(-1,1)[idx])
        
        u_pde = u_t + advection_z * u_z  + diffusion_z * u_zz  - linearTerm * u
        return u_pde
    
    def PDESolver(self,value_function_e,value_function_h,j0_e,j0_h,X,X_pde,idx,idx_crisis):
        fpde_e = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'experts')
        
        jE = value_function_e(tf.concat([X[:,0:1],X[:,1:2]],axis=1))
        
        
        fpde_h = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'households')
        jH = value_function_h(tf.concat([X[:,0:1],X[:,1:2]],axis=1))
        
        #experts
        loss_1_1 = fpde_e
        loss_1 = tf.reduce_mean(tf.square(loss_1_1)) 
        
        loss_2 = tf.reduce_mean(tf.square(jE - j0_e)) 

        #households
        loss_3 = tf.reduce_mean(tf.square(jH - j0_h)) 
        loss_4 = tf.reduce_mean(tf.square(fpde_h))

        
        loss = loss_1 + loss_2 + loss_3 + loss_4 
        return loss,fpde_e

    def loss_function(self,batchSize):
        idx = np.random.choice(self.X.shape[0],batchSize,replace=True)
        loss_total,fpde = self.PDESolver(self.value_function_e,self.value_function_h,self.J0_e,self.J0_h,self.X,self.X_pde,idx,idx)
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
            self.value_function_e = self.NN(2,self.params['num_layers'])
        
        if os.path.isfile('./save/value_function_households.h5'):
            self.value_function_h = load_model('./save/value_function_households.h5')
        else:
            self.value_function_h = self.NN(2,self.params['num_layers'])

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
        je_new = self.value_function_e(tf.concat([x_star[:,0:1],x_star[:,1:2]],axis=1))
        jh_new = self.value_function_h(tf.concat([x_star[:,0:1],x_star[:,1:2]],axis=1))
        return je_new.numpy(),jh_new.numpy()


class model_nnpde_B1():
    def __init__(self, params):
        self.params = params
        # algorithm parameters
        self.convergenceCriterion = 1e-5; 
        self.dt = 2; #time step width
        self.converged = 'False'
        self.Iter=0
        # grid parameters
        self.Nf = 1
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.999;
       
        # grid parameters
        self.Nz = 1000;
        zMin = 0.001; 
        zMax = 0.999;
        
        
        self.z = np.linspace(zMin,zMax,self.Nz);
        self.z_mat = np.tile(self.z,(1,1)).transpose()
        self.dz  = self.z_mat[1:self.Nz] - self.z_mat[0:self.Nz-1]; 
        self.dz2 = self.dz[0:self.Nz-2]**2
        self.dz_mat = np.tile(self.dz,(self.Nf,1))

        ## initial guesses [terminal conditions]
        self.Je = np.ones([self.Nz]) 
        self.Jh = np.ones([self.Nz]) 
        self.q = np.ones([self.Nz,1]);
        self.qz  = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        self.qzz = np.array(np.tile(0,(self.Nz,self.Nf)), dtype=np.float64); 
        
        # allocate memory for other variables
        self.psi = np.full([self.Nz,1],np.NaN)
        self.chi = np.full([self.Nz,1],np.NaN)
        self.ssq = np.full([self.Nz,1],np.NaN)
        self.iota = np.full([self.Nz,1],np.NaN)
        self.dq = np.full([self.Nz,1],np.NaN)
        self.amax=np.float('Inf')
        self.amax_vec = []

        #set up grid for pde
        self.X = np.vstack((self.z,np.full(self.z.shape[0],self.dt))).transpose().astype(np.float64)
        self.X_pde = np.vstack((self.z,np.random.uniform(0,self.dt,self.z.shape[0]))).transpose().astype(np.float64)
        self.x_star = np.vstack((self.z,np.full(self.z.shape[0],0))).transpose()
        

    def equations_region1(self, q_p, Psi_p, sig_ka_p, zi):  
        '''
        Solves for the equilibrium policy in the crisis region 
        Input: old values of capital price(q_p), capital share(Psi_p), return volatility(sig_ka_p), grid point(zi)
        Output: new values from Newton-Rhaphson method
        ''' 
        dz  = self.z[1:self.Nz] - self.z[0:self.Nz-1];  
        i_p = (q_p -1)/self.params['kappa']
        eq1 = (self.params['aE']-self.params['aH'])/q_p  - \
                            self.params['alpha']* (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * (self.params['alpha']* Psi_p - self.z[zi]) * sig_ka_p**2 - (self.params['gammaE'] - self.params['gammaH']) * sig_ka_p * self.params['sigma']
                    
        eq2 = (self.params['rhoE']*self.z[zi] + self.params['rhoH']*(1-self.z[zi])) * q_p  - Psi_p * (self.params['aE'] - i_p) - (1- Psi_p) * (self.params['aH'] - i_p)
        
        eq3 = sig_ka_p*(1-((q_p - self.q[zi-1][0])/(dz[zi-1]*q_p) * self.z[zi-1] *(self.params['alpha']* Psi_p/self.z[zi]-1)))  - self.params['sigma'] 
        ER = np.array([eq1, eq2, eq3])
        QN = np.zeros(shape=(3,3))
        QN[0,:] = np.array([-self.params['alpha']**2 * (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p**2, \
                            -2*self.params['alpha']*(self.params['alpha'] * Psi_p - self.z[zi]) * (self.dLogJh[zi]*(1-self.params['gammaH']) - self.dLogJe[zi]*(1-self.params['gammaE']) + 1/(self.z[zi] * (1-self.z[zi]))) * sig_ka_p + (self.params['gammaH'] - self.params['gammaE']) * self.params['sigma'], \
                                  -(self.params['aE'] - self.params['aH'])/q_p**2])
        QN[1,:] = np.array([self.params['aH'] - self.params['aE'], 0, self.z[zi]* self.params['rhoE'] + (1-self.z[zi])* self.params['rhoH'] + 1/self.params['kappa']])
        
        QN[2,:] = np.array([-sig_ka_p * self.params['alpha'] * (1- self.q[zi-1][0]/q_p) / (dz[zi-1]) , \
                          1 - (1- (self.q[zi-1][0]/q_p)) / dz[zi-1] * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1] , \
                            sig_ka_p * (-self.q[zi-1][0]/(q_p**2 * dz[zi-1]) * (self.params['alpha'] * Psi_p/self.z[zi] -1) * self.z[zi-1])])
        EN = np.array([Psi_p, sig_ka_p, q_p]) - np.linalg.solve(QN,ER)
        
        del ER
        del QN
        return EN

    def pickle_stuff(self,object_name,filename):
                    with open(filename,'wb') as f:
                        dill.dump(object_name,f)  
    def solve(self,pde=True):
        # initialize variables at z=0
        self.psi[0] = 0;
        self.q[0] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[0]*(self.params['aE']-self.params['aH'])))/(1 + self.params['kappa']*(self.params['rhoH'] + self.z[0] * (self.params['rhoE'] - self.params['rhoH'])));
        self.chi[0] = 0;
        self.ssq[0] = self.params['sigma'];
        self.q0 = (1 + self.params['kappa'] * self.params['aH'])/(1 + self.params['kappa'] * self.params['rhoH']); #heoretical limit at z=0 [just used in a special case below that is probably never entered]
        self.iota[0] = (self.q0-1)/self.params['kappa']
        
        for timeStep in range(self.params['maxIterations']):
            self.Iter+=1
            self.crisis_eta = 0;
            self.logValueE = np.log(self.Je);
            self.logValueH = np.log(self.Jh);
            self.dLogJe = np.hstack([(self.logValueE[1]-self.logValueE[0])/(self.z[1]-self.z[0]),(self.logValueE[2:]-self.logValueE[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueE[-1]-self.logValueE[-2])/(self.z[-1]-self.z[-2])]);
            self.dLogJh = np.hstack([(self.logValueH[1]-self.logValueH[0])/(self.z[1]-self.z[0]),(self.logValueH[2:]-self.logValueH[0:-2])/(self.z[2:]-self.z[0:-2]),(self.logValueH[-1]-self.logValueH[-2])/(self.z[-1]-self.z[-2])]);
            
            for i in range(1,self.Nz):
                  if self.psi[i-1] >= 1:
                          break; 
                  result= self.equations_region1(self.q[i-1][0], self.psi[i-1][0], self.ssq[i-1][0], i)
                  self.psi[i], self.ssq[i], self.q[i] =result[0], result[1], result[2]
                  self.chi[i] = self.params['alpha']*self.psi[i]
                  self.dq[i] = (1 - self.params['sigma']/self.ssq[i])/(self.chi[i] - self.z[i])*self.q[i]
                  self.iota[i] = (self.q[i]-1)/self.params['kappa']
            self.thresholdIndex = i-1;
            self.crisis_eta = self.z[self.thresholdIndex]
            self.crisis_flag = np.array(np.tile(0,(self.Nz,1)), dtype = np.float64)
            self.crisis_flag[0:self.thresholdIndex] = 1
            self.psi[self.thresholdIndex:] = 1;
            self.q[self.thresholdIndex:] = (1 + self.params['kappa']*(self.params['aH'] + self.psi[self.thresholdIndex:]*(self.params['aE']-self.params['aH']))).reshape(-1,1)/(1 + self.params['kappa']*(self.params['rhoH'] + self.z[self.thresholdIndex:]*(self.params['rhoE']-self.params['rhoH']))).reshape(-1,1);
            self.chi[self.thresholdIndex:] = np.maximum(self.z[self.thresholdIndex:],self.params['alpha']).reshape(-1,1); #NOTE: this seems incorrect for gammaE~=gammaH!
            #self.iota[self.thresholdIndex:] = 1 + self.params['kappa']* self.q[self.thresholdIndex:];
            self.iota[self.thresholdIndex:] = (self.q[self.thresholdIndex:]-1)/self.params['kappa']
            if self.thresholdIndex==0:
                self.dq[self.thresholdIndex:-1] = (self.q[1:] - np.vstack([self.q0,self.q[0:-2]])) / (self.z - np.vstack([0,self.z[:-2]])) #needs fixing
            else:
                self.dq[self.thresholdIndex:] = (self.q[self.thresholdIndex:]- self.q[self.thresholdIndex-1:-1]).reshape(-1,1)/(self.z[self.thresholdIndex:]-self.z[self.thresholdIndex-1:-1]).reshape(-1,1);
            self.ssq[self.thresholdIndex:] = self.params['sigma']/(1-self.dq[self.thresholdIndex:]/self.q[self.thresholdIndex:] * (self.chi[self.thresholdIndex:]-self.z[self.thresholdIndex:].reshape(-1,1)));
            self.theta = (self.chi)/self.z.reshape(-1,1)
            self.thetah = (1-self.chi)/(1-self.z.reshape(-1,1))
            self.theta[0] = self.theta[1]
            self.thetah[0] = self.thetah[1]
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.qz[1:self.Nz,:]  = (self.q [1:self.Nz,:] - self.q [0:self.Nz-1,:])/self.dz_mat; self.qz[0,:]=self.qz[1,:];
            self.qzz[2:self.Nz,:] = (self.q[2:self.Nz,:] + self.q[0:self.Nz-2,:] - 2.*self.q[1:self.Nz-1,:])/(self.dz2.reshape(-1,1)); self.qzz[0,:]=self.qzz[2,:]; self.qzz[1,:]=self.qzz[2,:]; 
            self.qzl  = self.qz/self.q ; 
            self.qzzl = self.qzz/ self.q;
            self.consWealthRatioE = self.params['rhoE'];
            self.consWealthRatioH = self.params['rhoH'];
            self.sig_za = (self.chi - self.z.reshape(-1,1))*self.ssq; #sig_za := \sigma^\z \z, similary mu_z
            self.priceOfRiskE = (1/self.z.reshape(-1,1) - self.dLogJe.reshape(-1,1)*(1-self.params['gammaE'])) * self.sig_za + self.ssq + (self.params['gammaE']-1) * self.params['sigma'];
            self.priceOfRiskH = -(1/(1-self.z.reshape(-1,1)) + self.dLogJh.reshape(-1,1)*(1-self.params['gammaH']))*self.sig_za + self.ssq + (self.params['gammaH']-1) * self.params['sigma'];
            self.sig_je = self.dLogJe.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.sig_jh = self.dLogJh.reshape(-1,1) * (self.sig_za.reshape(-1,1))
            self.rp = self.priceOfRiskE * self.ssq
            self.rp_ = self.priceOfRiskH * self.ssq
            self.mu_z = self.z_mat*((self.params['aE']-self.iota)/self.q - self.consWealthRatioE + (self.theta-1)*self.ssq*(self.rp/self.ssq - self.ssq) + self.ssq*(1-self.params['alpha'])*(self.rp/self.ssq - self.rp_/self.ssq) + (self.params['lambda_d']/self.z_mat)*(self.params['zbar']-self.z_mat)) 
            self.growthRate = np.log(self.q)/self.params['kappa']-self.params['delta'];
            self.sig_za[0] = 0; 
            self.mu_z[0] = 0
            self.Phi = (np.log(self.q))/self.params['kappa']
            self.mu_q = self.qzl * self.mu_z + 0.5*self.qzzl*self.sig_za**2 
            self.mu_rH = (self.params['aH'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.mu_rE = (self.params['aE'] - self.iota)/self.q + self.Phi - self.params['delta'] + self.mu_q + self.params['sigma'] * (self.ssq - self.params['sigma'])
            self.r = self.mu_rE - self.ssq*self.priceOfRiskE 
            self.A = self.psi*self.params['aE'] + (1-self.psi)*(self.params['aH'])
            self.AminusIota = self.psi*(self.params['aE'] - self.iota) + (1-self.psi) * (self.params['aH'] - self.iota)
            self.pd = np.log(self.q / self.AminusIota)
            
            #fix numerical issues
            self.r[self.thresholdIndex:self.thresholdIndex+2] = 0.5*(self.r[self.thresholdIndex+2] + self.r[self.thresholdIndex-1]) #r is not defined at the kink, so replace with average of neighbours to avoid numerical issues during simulation                     
   
            
            #########################################################################################
            #########################################################################################
            #########################################################################################
            # PDE time steps
                   
            if pde==True:
                self.Jhat_e = self.Je.copy().reshape(-1,1)
                self.Jhat_h = self.Jh.copy().reshape(-1,1)
                
                learning_rate = 0.001
                layers = self.params['num_layers']
                Jhat_e0 = self.Jhat_e.transpose().flatten().reshape(-1,1)
                Jhat_h0 = self.Jhat_h.transpose().flatten().reshape(-1,1)
                self.diffusion = self.sig_za**2/2;
                self.linearTermE =  (1-self.params['gammaE'])*self.params['sigma']*self.sig_je + 0.5*self.params['gammaE']*(self.sig_je**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoE']*(np.log(self.params['rhoE']) - np.log(self.Je.reshape(-1,1)) + np.log(self.z.reshape(-1,1)*self.q)) 
                self.linearTermH = (1-self.params['gammaH'])*self.params['sigma']*self.sig_jh + 0.5*self.params['gammaH']*(self.sig_jh**2 + self.params['sigma']**2)- self.growthRate - self.params['rhoH']*(np.log(self.params['rhoH']) - np.log(self.Jh.reshape(-1,1)) + np.log((1-self.z.reshape(-1,1))*self.q)) #PDE coefficient multiplying Jh
                self.advection = self.mu_z 

                #sovle the PDE
                print('Solving for value functions')
                model = nnpde_informed_B1(self.linearTermE.transpose().reshape(-1,1), self.advection.transpose().reshape(-1,1), 
                                         self.diffusion.transpose().reshape(-1,1),Jhat_e0.reshape(-1,1).astype(np.float64),self.z,layers,self.params['batchSize'],learning_rate,int(self.params['epochE']/np.sqrt(timeStep+1)),self.dt,
                                         self.linearTermH.transpose().reshape(-1,1), self.advection.transpose().reshape(-1,1),Jhat_h0.reshape(-1,1).astype(np.float64),self.params,self.X,self.X_pde)
                model.train()
                newJe,newJh = model.predict(self.x_star)
                model.value_function_e.save('./save/value_function_experts.h5')
                model.value_function_h.save('./save/value_function_households.h5')
                self.fpde_e = model.fpde
                del model 
                
                #upate Je and Jh for next static step iteration
                
                self.ChangeJe = np.abs(newJe.reshape(-1) - self.Je.reshape(-1))
                self.ChangeJh = np.abs(newJh.reshape(-1) - self.Jh.reshape(-1))
                self.relChangeJe = np.abs((newJe - self.Je) / self.Je)
                self.relChangeJh = np.abs((newJh- self.Jh) / self.Jh)
                #break if nan values
                if np.sum(np.isnan(newJe))>0 or np.sum(np.isnan(newJh))>0:
                    print('NaN values found in Value function')
                    break
                
                self.Jh = newJh.reshape(-1)
                self.Je = newJe.reshape(-1)
                
                if self.params['write_pickle']==True:
                    self.pickle_stuff(self,'modelB1' + '.pkl') 
                
                self.amax = np.maximum(np.amax(self.ChangeJe),np.amax(self.ChangeJh))
                
                if self.amax < self.convergenceCriterion:
                    self.converged = 'True'
                    break
                elif len(self.amax_vec)>1 and np.abs(self.amax - self.amax_vec[-1])>0.5:
                    print('check inner loop. amax error is very large: ',self.amax)
                    break
                print('Iteration number and Absolute max of relative error: ',self.Iter,',',self.amax)
                print('/n')
                self.amax_vec.append(self.amax)
                
    def clear_saved_models(self):            
                files = glob.glob('./save/*')
                for f in files:
                    os.remove(f)
                           
if __name__ =="__main__":
    params = {'rhoE': 0.05, 'rhoH': 0.05, 'aH': 0.02,'aE':0.1,
            'alpha':1.0, 'kappa':5, 'delta':0.05, 'zbar':0.1,
            'lambda_d':0.02, 'sigma':0.06, 'gammaE':2, 'gammaH':2, 
             'epochE':6000,'batchSize':500,
            'DGM':False,'num_neurons':256,'num_layers':4}
    params['write_pickle'] = args.save
    params['batchSize'] = args.batchSize
    ext = model_nnpde_B1(params)
    ext.params['maxIterations'] = args.maxIterations
    ext.solve(pde=True)
    ext.clear_saved_models()
    
    