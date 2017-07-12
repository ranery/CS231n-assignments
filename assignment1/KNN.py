# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:47:29 2017

@author: dell-pc

function: k-Nearest Neighbor Classifier

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class KNearestNeighbor (object):
    """ a KNN Classifier with L2 distance """
    def __init__ (self):
        
        pass
    
    def train (self, X, y):
        
        self.X_train = X
        self.Y_train = y
        
    def predict (self, X, k = 1, num_loops = 0):
        
        # get the L2 distance between X and X_train 
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        
        print('distant shape:')
        print(dists.shape)
        # plt.imshow(dists, interpolation = 'none')
        # plt.show()
            
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            min_index = np.argsort(dists[i])[0:k]
            closest_y = []
            closest_y = self.Y_train[min_index]
            a = stats.mode(closest_y)
            y_pred[i] = a[0]
            
        return y_pred
            
    def compute_distances_two_loops(self, X):
    
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          for j in range(num_train):
            dists[i,j] = np.sqrt(np.sum(np.square(np.abs(X[i] - self.X_train[j]))))

        return dists

    def compute_distances_one_loop(self, X):
    
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis = 1))
          # axis = 0 列相加； axis = 1 行相加 转置 => 行
    
        return dists
    
    def compute_distances_no_loops(self, X):
          
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        a = np.sum(np.square(X), axis = 1)
        b = np.sum(np.square(self.X_train), axis = 1)
        dists = np.sqrt(a.reshape(num_test, 1) + b - 2 * X.dot(self.X_train.T))
        
        return dists