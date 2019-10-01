# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:54:52 2019

@author: Uchiha Madara.
"""
#importing the libraries.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.autograd as variable
from sklearn.model_selection import train_test_split
from RBM import * #self made Restricted Boltzmann Machines class.

from hyperopt import *

#importing the dataset and specifying some required variables.
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t').values
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t').values
test_set, val_set = train_test_split(test_set, test_size = 0.5, random_state = 42)
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting the dataset to plausible format. (each row = user, each column movie = -1 if not rated, 0 if disliked, 1 if liked.)
def convert(data):
    l0 = []
    for user in range(nb_users):
        l1 = np.zeros(nb_movies)
        usrs_movies = data[:,1][data[:,0] == user]
        l1[usrs_movies-1] = data[:,2][data[:,0] == user]
        l0.append(list(l1))
    l0 = torch.FloatTensor(l0)
    data[data == 0] = -1
    data[data == 1] = 0
    data[data == 2] = 0
    data[data == 3] = 1
    data[data >= 4 ] = 1    
    return l0

#converting our sets according to given function.
training_set = convert(training_set)
test_set = convert(test_set)
val_set = convert(val_set)
    
#Making our RBM and tuning hyperparameters.

#Description of Hparams taken by the train_rbm:
#nv = nb_movies
#nh = 200 (200 features to learn).
#batch_size = 128 (no. of training examples to take for batch learning).
#epochs = 20 (no. of iterations through the training set).
#rbm = RBM(nv, nh) (Object declaration of the RBM class).

#Function to get accuracy on the cross-validation set of the dataset.
def validate_rbm(rbm):    
    test_loss_mae = 0.0
    s = 0.0
    for user in range(nb_users):
        v = training_set[user:user+1]
        v0 = test_set[user:user+1]
        if(len(v0[v0>=0]) > 0):
            _,h = rbm.sample_h(v)
            _,v = rbm.sample_v(h)
            test_loss_mae += torch.mean(torch.abs(v0[v0 >=0 ] - v[v0 >=0]))
            s+=1.0
    return float(test_loss_mae/s)

#Function to train the RBM based on given Hparams.
def train_rbm(nv, nh, batch_size, epochs,val_set, gibbs_sampling_nb, rbm = None):
    if(rbm == None):
        rbm = RBM(nv, nh)
    for epoch in range(1, epochs+1):
        training_loss = 0
        s = 0.0
        for user in range(0, nb_users-batch_size, batch_size):
            vk = val_set[user:user+batch_size]
            v0 = val_set[user:user+batch_size]
            for sample in range(gibbs_sampling_nb):
                _,hk = rbm.sample_h(vk)
                _,vk = rbm.sample_v(hk)
                vk[v0 <0] = v0[v0 <0]
                phk,_ = rbm.sample_h(vk)
                ph0,_ = rbm.sample_h(v0)
                rbm.train(v0,vk,ph0,phk)
            training_loss += torch.mean(torch.abs(v0[v0 >=0 ] - vk[v0 >=0]))
            s+=1
       # print("Epoch:", epoch, "Training Loss:", training_loss/s)
    return float(training_loss/s),rbm

#Defining the Hparam space.
space = {
            'nh' : hp.choice('nh', [int (x) for x in range(100,500,50)]),
            'batch_size' : hp.choice('batch_size',[int (x) for x in range(32, 256, 16)]),
            'epochs' : hp.choice('epochs', [int (x) for x in range(10,50,10)]),
            'gibbs_sampling_nb' : hp.choice( 'gibbs_sampling_nb', [int (x) for x in range(5,30,5)])
        }

#The function to optimize by training on specific hparams and then getting the optimization 
#cost by validate_rbm() function.

def RBM_opt_fn(space):
    nv = nb_movies
    nh = space['nh']
    batch_size = space['batch_size']
    epochs = space['epochs']
    gibbs_sampling_nb = space['gibbs_sampling_nb']
    val_train,rbm = train_rbm(nv, nh, batch_size, epochs,training_set, gibbs_sampling_nb)
    val = validate_rbm(rbm)
    print('Hyperopt Loss:',val, 'nh:', nh, 'batsz:',batch_size, 'ep:',epochs, 'gsnb:', gibbs_sampling_nb )
    return{'loss' : val , 'status' : STATUS_OK}

#getting the best params using hyperopt class.
trials = Trials()
best = fmin(
                    fn = RBM_opt_fn,
                    space = space,
                     algo = tpe.suggest,   
                    max_evals = 100,
                    trials = trials
                )
print(best)  
#best params :  nh = 100, bs = 80, epochs = 40, gsnb = 5. 
#    training_loss = 0
#    s = 0.0
#    for user in range(0, nb_users-batch_size, batch_size):
#        vk = training_set[user:user+batch_size]
#        v0 = training_set[user:user+batch_size]
#        for sample in range(gibbs_sampling_nb):
#            _,hk = rbm.sample_h(vk)
#            _,vk = rbm.sample_v(hk)
#            vk[v0 <0] = v0[v0 <0]
#        phk,_ = rbm.sample_h(vk)
#        ph0,_ = rbm.sample_h(v0)
#        rbm.train(v0,vk,ph0,phk)
#        training_loss += torch.mean(torch.abs(v0[v0 >=0 ] - vk[v0 >=0]))
#        s+=1
#    print("Epoch:", epoch, "Training Loss:", training_loss/s)

#Testing the RBM against the test set.
test_loss_mae = 0.0
s = 0.0
for user in range(nb_users):
    v = training_set[user:user+1]
    v0 = test_set[user:user+1]
    if(len(v0[v0>=0]) > 0):
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss_mae += torch.mean(torch.abs(v0[v0 >=0 ] - v[v0 >=0]))
        s+=1.0
print("Test Loss Mean Absolute Error:", test_loss_mae/s)