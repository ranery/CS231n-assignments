# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:24:50 2017

@author: dell-pc

function: image classification by using SVM

"""

import numpy as np
import matplotlib.pyplot as plt

from load_data import X_train, Y_train, X_test, Y_test

# split the data into train/validation/test and a small development set 
num_training = 4900
num_validation = 100
num_test = 100
num_dev = 50
# validation set
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
Y_val = Y_train[mask]
# training set
mask = range(num_training)
X_train = X_train[mask]
Y_train = Y_train[mask]
# development set (small subset of the training set)
mask = np.random.choice(num_training, num_dev, replace = False)
X_dev = X_train[mask]
Y_dev = Y_train[mask]
# test set
mask = range(num_test)
X_test = X_test[mask]
Y_test = Y_test[mask]

"""
preprocessing : reshape the image data into raws

"""
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape:     ', X_train.shape)
print('Train labels shape:      ', Y_train.shape)
print('Validation data shape:   ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape:         ', X_test.shape)
print('Test labels shape:       ', Y_test.shape)
print('dev data shape:          ', X_dev.shape)
print('dev labels shape:        ', Y_dev.shape)

"""
preprocessing : zero-mean

"""
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis = 0)
plt.figure(figsize = (4, 4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

"""
SVM Classifier

"""
from SVM import svm_loss_naive
import time
W = np.random.randn(3073, 10) * 0.0001
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, Y_dev, 0.00001)
toc = time.time()
print('Naive loss: %e computed in %fs' %(loss_naive, toc - tic))

from SVM import svm_loss_vectorized
tic = time.time()
loss_vec, grad_vec = svm_loss_vectorized(W, X_dev, Y_dev, 0.00001)
toc = time.time()
print('Vectorized loss: %e computed in %fs' %(loss_vec, toc - tic))

print('loss difference: %f' %(loss_naive - loss_vec))
print('gradient difference: %f' %(np.linalg.norm(grad_naive - grad_vec, ord='fro')))

# check the gradient
from gradient_check import grad_check_sparse
f = lambda w: svm_loss_vectorized(w, X_dev, Y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad_vec)

loss_vec, grad_vec = svm_loss_vectorized(W, X_dev, Y_dev, 1e2)
f = lambda w: svm_loss_vectorized(w, X_dev, Y_dev, 1e2)[0]
grad_numerical = grad_check_sparse(f, W, grad_vec)

# implement SGD 
from linear_classifier import LinearSVM
svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, Y_train, learning_rate = 1e-7, reg = 5e4,
                      num_iter = 1500, verbose = True)
toc = time.time()
print('That took %fs' %(toc - tic))

# loss --- iterations
plt.figure()
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# predict
y_train_pred = svm.predict(X_train)
print('training set accuracy: %f' %(np.mean(Y_train == y_train_pred)))
y_val_pred = svm.predict(X_val)
print('validation set accuracy: %f' %(np.mean(Y_val == y_val_pred)))

# find best
learning_rates = [10 ** x for x in range(-7, 7)]
reg_strength = [10 ** x for x in range(-7, 7)]

results = {}
best_val = -1
best_svm = None
for lr in learning_rates:
    for reg in reg_strength:
        svm = LinearSVM()
        svm.train(X_train, Y_train, learning_rate = lr, reg = reg, num_iter = 1500)
        
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        train_acc = np.mean(y_train_pred == Y_train)
        val_acc = np.mean(y_val_pred == Y_val)
        
        results[(lr, reg)] = (train_acc, val_acc)
        
        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm
            
for lr, reg in sorted(results):
    train_acc, val_acc = results[(lr, reg)]
    print('lr: %e reg: %e taining accuracy: %f val accuracy: %f' %(
                lr, reg, train_acc, val_acc))
    
print ('the best validation accuracy achieved during cross-validation: %f' % best_val)

# visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]  # learning rate
y_scatter = [math.log10(x[1]) for x in results]  # regularization strengths

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.figure()
plt.subplot(2,1,1)
plt.scatter(x_scatter, y_scatter, marker_size, c = colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results]
plt.subplot(2,1,2)
plt.scatter(x_scatter, y_scatter, marker_size, c = colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')


# evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_acc = np.mean(y_test_pred == Y_test)
print('linear SVM on test set final accuracy: %f' %(test_acc))

# visualize the learned weighted for each class
w = best_svm.W[:-1,:] # strip out bias
w = w.reshape(32,32,3,10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    wimg = 255 * (w[:,:,:,i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])