# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:50:55 2017

@author: Haoran You

function: neural_net

"""

import numpy as np

class TwoLayerNet(object):
    """
    two layer neural network
    
    """
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # sigmoid function
    def nonlin (self,X):
        return np.maximum(0, X)
        
    def loss(self, X, y = None, reg = 0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        
        # forward
        l1 = X.dot(W1) + b1
        l1 = self.nonlin(l1)
        l2 = l1.dot(W2) + b2
        scores = l2

        
        if y is None:
            return scores
            
        # compute svm loss
        correct_class_scores = scores[range(N), y].reshape((N, 1))
        margin = np.maximum(0.0, scores - correct_class_scores + 1.0)
        margin[range(N), y] = 0.0
    
        loss = np.sum(margin) / N 
        loss = loss + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        # compute svm gradient
        grad = {}
        B = np.full((N, 1), 1)
        margin[margin > 0] = 1
        margin[range(N), y] = -1 * np.sum(margin, axis = 1)
        grad['W2'] = l1.T.dot(margin) / N + W2 * reg
        grad['b2'] = B.T.dot(margin)
        
        margin_hide = margin.dot(W2.T) * (l1 > 0)
        grad['W1'] = X.T.dot(margin_hide) / N + W1 * reg
        grad['b1'] = B.T.dot(margin_hide)
        
        return loss, grad
        
    def train(self, X, y, X_val, y_val, learning_rate = 1e-3, learning_rate_decay = 0.95,
              reg = 1e-5, num_iter = 100, batch_size = 100, verbose = False):
        """
        using SGD
        
        """
        num_train = X.shape[0]
        iter_per_epoch = max(num_train / batch_size, 1)
        
        # SGD
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iter):
            
            batch_ix = np.random.choice(num_train, batch_size) 
            X_batch = X[batch_ix] 
            y_batch = y[batch_ix]
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            
            for w in self.params:
                self.params[w] -= grad[w].reshape(self.params[w].shape) * learning_rate

            if verbose and it % 100 == 0:
                print('iteration: %d / %d  loss : %f' %(it, num_iter, loss))
                
            # each epoch, check train and val accuracy and decay learning rate
            if it % iter_per_epoch == 0:
                # check accuracy
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_batch)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                # Decay learning rate
                learning_rate *= learning_rate_decay
                
        return{
               'loss_history': loss_history,
               'train_acc_history': train_acc_history,
               'val_acc_history': val_acc_history,
               }
               
    def predict(self, X):
        """
        predict
        
        """
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis = 1)
        
        return y_pred

            
