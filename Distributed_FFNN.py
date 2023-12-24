#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:43:54 2021

@author: Mahmoud Zaher
"""

# This script is used for training the DDNN and DDNN-SI models for the different
# optimization objectives and precoding schemes considered.
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from os import makedirs
from sklearn import preprocessing
# import pandas as pd
import pathlib
from pickle import dump

base_path = pathlib.Path().absolute()
filename = str(base_path) + '\\pAssign_storage\\'
models_filename = str(base_path) + '\\pAssignModels\\DNNmodels-extrainput\\'
#print(str(base_path) + '\\pAssignModels\\DNNmodels\\')
#reproducible results using Keras
sd = 42# Here sd means seed. 随机种子是为了确保实验结果能够被复现

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(sd)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random as rn
rn.seed(sd)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(sd)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(sd)

# 5. Configure a new global `tensorflow` session
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)


# Loading input to the NN
# 数据加载和预处理部分
# 
mu_MR_sumSE_DNN = np.load(filename + 'dataset_mu_MR_WMMSE_ADMM.npy',allow_pickle=True)  
mu_RZF_sumSE_DNN = np.load(filename + 'dataset_mu_RZF_WMMSE_ADMM.npy',allow_pickle=True)   
mu_MR_PF_DNN = np.load(filename + 'dataset_mu_MR_WMMSE_PF_ADMM.npy',allow_pickle=True)   
mu_RZF_PF_DNN = np.load(filename + 'dataset_mu_RZF_WMMSE_PF_ADMM.npy',allow_pickle=True)
betas_DNN = np.load(filename + 'dataset_betas.npy',allow_pickle=True)
# print("mu_MR_sumSE_DNN.shape:", mu_MR_sumSE_DNN.shape) # 20,16,350040
# print("mu_RZF_sumSE_DNN.shape:", mu_RZF_sumSE_DNN.shape) # 20,16,350040
# print("mu_MR_PF_DNN.shape:", mu_MR_PF_DNN.shape) # 20,16,350040
# print("mu_RZF_PF_DNN.shape:", mu_RZF_PF_DNN.shape) # 20,16,350040
# print("betas_DNN.shape:", betas_DNN.shape) # 20,16,350040
# print("mu_MR_sumSE_DNN[0,:,0]:", mu_MR_sumSE_DNN[0,:,0])
# print("mu_RZF_sumSE_DNN[0,:,0]:", mu_RZF_sumSE_DNN[0,:,0])
# print("mu_MR_PF_DNN[0,:,0]:", mu_MR_PF_DNN[0,:,0])
# print("mu_RZF_PF_DNN[0,:,0]:", mu_RZF_PF_DNN[0,:,0])
# print("betas_DNN[0,:,0]:", betas_DNN[0,:,0])

#Maximum downlink transmit power per BS (mW)
Pmax = 1000 # 最大下行传输功率，以 mW 为单位
K = betas_DNN.shape[0] # 用户数量20

L = betas_DNN.shape[1] # 接入点AP数量16

NoOfSetups = betas_DNN.shape[2] # 350040
# Make sure the sum over the K UEs gives Pmax for each AP in each setup- (might not be necessary)
# 基于 Pmax 对信道状态信息进行缩放，以确保每个接入点（AP）的功率总和不超过 Pmax。
for n in range(NoOfSetups):
    # print("np.power(mu_MR_sumSE_DNN[:,:,n],2).shape:", np.power(mu_MR_sumSE_DNN[:,:,n],2).shape)
    # print("np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0).shape:", np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0).shape)
    # print("np.max (np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0)).shape:", np.max (np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0) ).shape)
    # input()
    mu_MR_sumSE_DNN[:,:,n] = mu_MR_sumSE_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0) )) )
    mu_RZF_sumSE_DNN[:,:,n] = mu_RZF_sumSE_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_RZF_sumSE_DNN[:,:,n],2), axis=0) )) )
    mu_MR_PF_DNN[:,:,n] = mu_MR_PF_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_MR_PF_DNN[:,:,n],2), axis=0) )) )
    mu_RZF_PF_DNN[:,:,n] = mu_RZF_PF_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_RZF_PF_DNN[:,:,n],2), axis=0) )) )

# [0,0,0]的平方 + [0,1,0]的平方 + 。。。 =[0,:,0]
# Maximum number of epochsS
Num_epoch = 20
# Batch size
N_batch_size = 128
K_initializer = 'random_normal'
B_initializer = 'random_normal'


##########################     MODEL FFNN   ##########################

# Optimizer
adam = tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #Decay = 0.1 not working

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0., patience=20, verbose=1, mode='auto')
callback = [early_stopping]

## Adjust the layers for DDNN or DDNN-SI model as per the tables in the paper
for l in range(0, L):
    model = Sequential()
    model.add(Dense(64, input_dim=2*K, activation="linear", name = 'layer1', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(128, input_dim=64, activation="elu", name = 'layer2', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(64, input_dim=128, activation="tanh", name = 'layer3', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(32, input_dim=64, activation="tanh", name = 'layer4', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(K+1, input_dim=32, activation="relu", name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))

    #Preparing inputs for NN
    #beta vector preparation(removing outliers and scaling)
    #预处理，移除异常值并缩放
    betas = betas_DNN[:,l,:].T  # or use a function of betas
    betas = 10*np.log10(betas*1000) # dB scale 将betas以分贝为单位进行处理
    big_values = []
    for i in range(0, NoOfSetups): # 如果任何一个数据点在 betas 中的值大于 34（以分贝为单位），则将该系统设置的索引添加到 big_values 列表中。
        if np.any(betas[i,:] > 34): # (i,20)20个数中任意一个大于34，就进行处理
            big_values = big_values + [i]
    betas = np.delete(betas, big_values, axis=0)
    #print("betas.shape:", betas.shape) # 315749, 20
    #print("betas[x]:", betas[x,:]) # 350040，20
    #input()
    NoOfSetups = betas.shape[0]
    
#########################################################################3
    # without注释
    scaler = preprocessing.RobustScaler(
                                with_centering=False,
                                with_scaling=True,
                                quantile_range=(25.0, 75.0),
                                copy=True,)
    
    betas = 10**(betas/10)      # changing back to linear scale 转换回线性单位
#########################################################################3



#########################################################################3
    ## For calculation of the extra input used in the DDNN-SI model
    # 当不需要DDNN-SI时注释
    v = 0.6                     # Fractional power allocation factor
    betas_to_all = np.delete(betas_DNN, big_values, axis = 2) * 1000    # linear scale (no kilo)
    extraInput = np.sqrt(Pmax) * ((betas ** v) / np.sum(betas_to_all ** v, axis=1 ).T)
    extraInput = 10*np.log10(extraInput)    # dB scale
    
    # The betas are changed to mus with fractional power allocatrion (gives better scaling)
    # 将贝塔值更改为使用分数功率分配的μ值（提供更好的缩放）。
#########################################################################3

    v = 0.6                     # Fractional power allocation factor
    betas = np.sqrt(Pmax) * ((betas ** v) / np.sum( (betas ** v), axis=1 ).reshape(NoOfSetups,1))
    betas = 10*np.log10(betas)  # dB scale
    
    betas = scaler.fit_transform(betas) # without时注释


#########################################################################3
    ## The extra input used in the DDNN-SI model
    extraInput = scaler.transform(extraInput)
    print("extraInput.shape:", extraInput.shape)
    DNNinput = np.concatenate((betas, extraInput), axis=1) # np.concatenate((betas, extraInput), axis=1) # betas  # betas is for DDNN,Second option is for DDNN-SI
#########################################################################3
    print("DNNinput.shape:", DNNinput.shape)
    x_train = DNNinput[0:NoOfSetups-100,:]

    ### Choose one of the following 4 options to train the model for
    # mu_MR_sumSE_DNN, mu_RZF_sumSE_DNN, mu_MR_PF_DNN, mu_RZF_PF_DNN
    ## mu preparation
    mu = np.abs(mu_RZF_PF_DNN[:,l,:].T)
    mu = np.delete(mu, big_values, axis=0)
    
    temp = np.sqrt( np.reshape(sum((mu[0:NoOfSetups-100,:].T) ** 2), (NoOfSetups-100, 1)) / Pmax)
    y_train = np.concatenate((mu[0:NoOfSetups-100,:], temp), axis=1)
    small_values = []
    small_val = 5 / np.sqrt(Pmax)
    for i in range(0, NoOfSetups-100):
        if y_train[i,K] < small_val:
            small_values = small_values + [i]
        
    y_train = np.delete(y_train, small_values, axis=0)
    NoOfSetups = y_train.shape[0]
    y_train[y_train < 0.001] = 0.001
    y_train[:,0:K] = np.sqrt(K) * tf.keras.utils.normalize(y_train[:,0:K], axis=1)
    #############################################################
    
    x_train = np.delete(x_train, small_values, axis=0)
    

    model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])
    
    print(model.summary())

    k.set_value(model.optimizer.lr, 0.001)
    history = model.fit(x_train, y_train, epochs = Num_epoch, batch_size = N_batch_size, validation_split = 0.1, callbacks=callback)
    k.set_value(model.optimizer.lr, 0.0001)
    history2 = model.fit(x_train, y_train, epochs = 5, batch_size = N_batch_size, validation_split = 0.1, callbacks=callback)

    x_test = DNNinput[NoOfSetups-100:NoOfSetups,:]
    ## Assign y_test based on the model choice above
    y_test = mu[NoOfSetups-100:NoOfSetups,:]
    y_test = np.sqrt(K) * tf.keras.utils.normalize(y_test, axis=1)
    y_predictions = model.predict(x_test)
    test_mse = np.mean((y_test - y_predictions[:,0:K])**2)
    print('Test MSE:' + str(test_mse))
    # Save models (You may adjust the name based on the chosen model)
    model.save(models_filename + "Trained_models_for_mu_RZF_PF_DDNN_for_AP" + str(l+1))
    print('Saved model %s' % models_filename + str(l+1))
# Save Scaler for betas (only required once)
dump(scaler, open(models_filename + 'scaler.pkl', 'wb'))