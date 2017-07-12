# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:58:51 2017

@author: Haoran You

function: load data

"""
import numpy as np
import os
import pickle

def load_Cifar_batch(filename):
    """ load single batch of cifar """
    with open (filename, 'rb') as f:
        datadict = pickle.load(f, encoding = 'latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_Cifar10(root):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(root, 'data_batch_%d' %(b, ))
        X, Y = load_Cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X,Y
    Xte, Yte = load_Cifar_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
    
def get_Cifar10_data(num_training = 49000, num_validation = 1000, num_test = 1000,
                     subtract_mean = True):
    """ load Cifar-10 data """
    cifar10_dir = '/home/ranery/Desktop/cs231n/assignment1/cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = load_Cifar10(cifar10_dir)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = Y_test[mask]

    # Normalize the data
    if subtract_mean:
        mean_image = np.mean(X_train, axis = 0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        
    # transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    
    # package data into a dictionary
    return {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'X_test': X_test, 
            'y_test': y_test,
            }
    
"""    
X_train, Y_train, X_test, Y_test = load_Cifar10('E:/Python/CS231n/assignment1/cifar-10-batches-py')


# Subsample the data for more efficient code execution
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
Y_train = Y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
Y_test = Y_test[mask]

"""
