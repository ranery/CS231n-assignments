# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 09:09:31 2017

@author: Haoran_You

function: softmax implementation

"""
import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function (naive implementation)
    
    """
    loss = 0
    dW = np.zeros_like(W)
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        sum_j = 0
        for j in range(num_classes):
            sum_j += np.exp(scores[j])
            
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) * X[i]) / sum_j
            if (j == y[i]):
                dW[:, j] -= X[i]

        loss += -correct_class_score + np.log(sum_j)
        
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_train
    dW += W * reg
    
    return loss, dW
    
def softmax_loss_vec(W, X, y, reg):
    """
    Softmax loss function (vectorized implementation)
    
    """
    loss = 0
    dW = np.zeros_like(W)
    
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_score = scores[range(num_train), y].reshape((num_train, 1))
    sum_j = np.sum(np.exp(scores), axis = 1).reshape((num_train, 1))
    
    loss = np.sum(-1 * correct_class_score + np.log(sum_j)) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    
    correct_matrix = np.zeros(scores.shape)
    correct_matrix[range(num_train), y] = 1
    
    dW = X.T.dot(np.exp(scores) / sum_j) - X.T.dot(correct_matrix)
    dW = dW / num_train + W * reg
    
    return loss, dW