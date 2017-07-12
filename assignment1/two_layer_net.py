# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:53:13 2017

@author: Haoran_You

function: two layer network

"""

import numpy as np
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet

# creat a small net

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

net = init_toy_model()
X, y = init_toy_data()

# compute scores
scores = net.loss(X)
print ('Your scores:')
print (scores)

# compute loss
loss, _ = net.loss(X, y, reg=0.1)
print('loss: %d'  %(loss))

# train
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iter=100, verbose=False)

print ('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

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

def show_net_graph(stats):
    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

# find best model
best_net = None 
input_size = 32 * 32 * 3
hidden_size = 100
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, Y_train, X_val, Y_val,
            num_iter=5000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.3, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == Y_val).mean()

# Summary
print ('Validation accuracy: ', val_acc)
show_net_graph(stats)

best_net = net

# test
test_acc = np.mean(best_net.predict(X_test) == Y_test)
print('Test accuracy: %f' %(test_acc) )
