# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:59:55 2017

@author: Haoran You

function: framework of classifier

"""

import numpy as np
from SVM import svm_loss_vectorized
from Softmax import softmax_loss_vec

class LinearClassifier (object):
    
    def __init__ (self):
        self.W = None
        
    def train(self, X, y, learning_rate = 1e-3, reg = 1e-5, num_iter = 100, 
              batch_size = 100, verbose = False):
        """
        training linear classifier using stochastic gradient desent
        
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assum y take values 0..K-1
        # initialize weight
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        
        # SGD
        loss_history = []
        for iter in range(num_iter):
            batch_ix = np.random.choice(num_train, batch_size)
            X_batch = X[batch_ix]
            y_batch = y[batch_ix]
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= grad * learning_rate
            
            if verbose and iter % 100 == 0:
                print('iteration: %d / %d: loss: %f' %(iter, num_iter, loss))
                
        return loss_history
        
    def predict(self, X):
        """
        predict labels for test set by using the training weight
        
        """
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(X.dot(self.W), axis = 1)
        
        return y_pred
            
    
class LinearSVM(LinearClassifier):
    """
    a subclass that uses the Multiclass SVM loss funtion
    
    """
    def loss (self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
       
        
class Softmax(LinearClassifier):
    """
    a sunclass that uses the Softmax and cross-entropy loss funtion
    
    """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vec(self.W, X_batch, y_batch, reg)
        
            
        