# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:25:35 2017

@author: dell-pc

function: support vector machine

"""

import numpy as np

def svm_loss_naive (W, X, y, reg):
    
    # initialize the gradient of W 
    dW = np.zeros(W.shape)
    # compute loss and gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    loss = 0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
          if j == y[i]:
            continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            loss += margin
            dW[:, j] += X[i] 
            dW[:, y[i]] -= X[i] 

    loss /= num_train
    dW /= num_train
    
    # add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)
    dW += W * reg
    
    return loss, dW
    
def svm_loss_vectorized (W, X, y, reg):
    """
    vectorized implementation
    
    """
    # initialize loss and the gradient
    loss  = 0
    dW = np.zeros(W.shape)
    
    # compute the loss and gradient
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = scores[range(num_train), y].reshape((num_train, 1))
    margin = np.maximum(0.0, scores - correct_class_score + 1.0)
    margin[range(num_train), y] = 0.0
    
    loss = np.sum(margin) / num_train + 0.5 * reg * np.sum(W * W)
    
    # compute the gradient 
    margin[margin > 0] = 1
    margin[range(num_train), y] = -1 * np.sum(margin, axis = 1)
    dW = X.T.dot(margin) / num_train + W * reg

    return loss, dW
    
    
    
    