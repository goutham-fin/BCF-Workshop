import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import time
import os


class nnpde_informed_B1():
    '''
    class to update the value function by solving a system of PDEs
    inputs: linear, advection, and diffusion terms for the PDEs. The index E/e and H/h refers to experts
    and households respectively.
    lr: learning rate
    adam_iter: epochs for adam loop
    J0_e and J0_h: initial values at time boundary for the expert and household value functions
    params: parameter dictionary
    X and X_pde: state space grid (can be gotten rid of for high dimensional problems)
    '''
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
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr) #set the optimizer
        self.initializer = tf.keras.initializers.GlorotNormal() #initialize the weights (important!)
        self.batchSize = batchSize 
        self.number_epochs = adam_iter
        self.lowest_iter= 0 
        self.min_loss = 20000 
        self.params=params

        
        self.X,self.X_pde = X,X_pde

    def NN(self,inputDim,num_layers,num_neurons=30):
        '''
        function that creates a fully-connected feed-forward neural network with tanh activation function in hidden layers
        '''
        model_ = keras.models.Sequential()
        model_.add(keras.layers.Dense(num_neurons,activation='tanh',input_dim=inputDim,kernel_initializer = self.initializer))
        for layer in range(num_layers-1):
            model_.add(keras.layers.Dense(num_neurons,activation='tanh',kernel_initializer=self.initializer))
        model_.add(keras.layers.Dense(1,kernel_initializer=self.initializer))
        return model_

    def get_value_pde(self,value_function_e,value_function_h,X_pde,idx,false_transient,agent_type):
        '''
        output: value function PDE residual
        '''
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

    def PDESolver(self,value_function_e,value_function_h,j0_e,j0_h,X,X_pde,idx):
        '''
        input: value functions as neural network objects, initial values, state space, and random sample id
        output: PDE residual and time boundary loss
        '''
        fpde_e = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'experts')
        
        jE = value_function_e(tf.concat([X[:,0:1],X[:,1:2]],axis=1))
        
        fpde_h = self.get_value_pde(value_function_e,value_function_h,X_pde,idx,True,'households')
        jH = value_function_h(tf.concat([X[:,0:1],X[:,1:2]],axis=1))
        
        #experts
        loss_1 = tf.reduce_mean(tf.square(fpde_e)) 
        loss_2 = tf.reduce_mean(tf.square(jE - j0_e)) 

        #households
        loss_3 = tf.reduce_mean(tf.square(jH - j0_h)) 
        loss_4 = tf.reduce_mean(tf.square(fpde_h))

        
        loss = loss_1 + loss_2 + loss_3 + loss_4 
        return loss,fpde_e

    def loss_function(self,batchSize):
        '''
        compute the PDE loss and boundary loss
        '''
        idx = np.random.choice(self.X.shape[0],batchSize,replace=True)
        loss_total,fpde = self.PDESolver(self.value_function_e,self.value_function_h,self.J0_e,self.J0_h,self.X,self.X_pde,idx)
        return loss_total,fpde

    @tf.function 
    def training_step(self):
        '''
        Function to gather the total loss
        '''
        with tf.GradientTape(persistent=True) as tape:
            loss_total,fpde= self.loss_function(self.batchSize)
        grads_value_e = tape.gradient(loss_total,self.theta_valueFunction_e)
        grads_value_h = tape.gradient(loss_total,self.theta_valueFunction_h)
        self.optimizer.apply_gradients(zip(grads_value_e,self.theta_valueFunction_e))
        self.optimizer.apply_gradients(zip(grads_value_h,self.theta_valueFunction_h))
        return loss_total,fpde

    def train(self):
        '''
        main function to train the model
        '''
        if os.path.isfile('./save/value_function_experts.h5'): #load the pre-trained model
            self.value_function_e = load_model('./save/value_function_experts.h5')
        else:
            self.value_function_e = self.NN(2,self.params['num_layers'],self.params['num_neurons'])
        
        if os.path.isfile('./save/value_function_households.h5'):
            self.value_function_h = load_model('./save/value_function_households.h5')
        else:
            self.value_function_h = self.NN(2,self.params['num_layers'],self.params['num_neurons'])

        self.theta_valueFunction_e = self.value_function_e.trainable_variables
        self.theta_valueFunction_h = self.value_function_h.trainable_variables
        
        self.best_valueFunction_e = tf.keras.models.clone_model(self.value_function_e)
        self.best_valueFunction_h = tf.keras.models.clone_model(self.value_function_h)

        self.LVF = []

        min_loss = float('inf')
        
        start_time = time.time()
        for epoch in range(self.number_epochs+1):
                loss_total,fpde = self.training_step()
                if (loss_total<min_loss): #update only when subsequent loss is smaller
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