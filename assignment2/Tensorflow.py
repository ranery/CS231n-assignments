#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:01:11 2017

@author: Haoran You

function : Tensorflow implementation

"""

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from load_data import load_Cifar10

########################### load Cifar-10 data ################################
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '/home/ranery/Desktop/cs231n/assignment1/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_Cifar10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

print('Train data shape:        \t', X_train.shape)
print('Train labels shape:      \t', y_train.shape)
print('Validation data shape:   \t', X_val.shape)
print('Validation labels shape: \t', y_val.shape)
print('Test data shape:         \t', X_test.shape)
print('Test labels shape:       \t', y_test.shape)

########################## run model function #################################
def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

"""
# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out

y_out = simple_model(X,y)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
        
# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape = [7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape = [32])
    W1 = tf.get_variable("W1", shape = [5408, 1024])
    b1 = tf.get_variable("b1", shape = [1024])
    W2 = tf.get_variable("W2", shape = [1024, 10] )
    b2 = tf.get_variable("b2", shape = [10])
   
    # define our graph
    # convelutional layer
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + bconv1
    # relu activation layer
    h1 = tf.nn.relu(a1)
    # spatial batch normalization
    bn1 = tf.layers.batch_normalization(h1, axis=1)
    # max pooling
    mp1 = tf.nn.max_pool(value = bn1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # affine layer
    mp1_flat = tf.reshape(mp1,[-1, 5408])
    a2 = tf.matmul(mp1_flat, W1) + b1
    # relu activation layer
    h2 = tf.nn.relu(a2)
    # affine layer
    y_out = tf.matmul(h2, W2) + b2
    return y_out

y_out = complex_model(X,y,is_training)

# Now we're going to feed a random batch into the model 
# and make sure the output is the right size
x = np.random.randn(64, 32, 32,3)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()

        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))
        
try:
    with tf.Session() as sess:
        with tf.device("/gpu:0") as dev: #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()

            ans = sess.run(y_out,feed_dict={X:x,is_training:True})

except tf.errors.InvalidArgumentError:
    print("no gpu found, please use Google Cloud if you want GPU acceleration")    
    # rebuild the graph
    # trying to start a GPU throws an exception 
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = complex_model(X,y,is_training)
    
"""
######################### Training my model ###################################

def my_model(X, y, is_training):
    # setup variable
    Wconv1 = tf.get_variable("Wconv1", shape = [3, 3, 3, 512])
    bconv1 = tf.get_variable("bconv1", shape = [512])
    # (32 - 3) / 1 + 1 = 30
    
    Wconv2 = tf.get_variable("Wconv2", shape = [3, 3, 512, 256])
    bconv2 = tf.get_variable("bconv2", shape = [256])
    # (15 - 3) / 1 + 1 = 13
    
    Wconv3 = tf.get_variable("Wconv3", shape = [3, 3, 256, 256])
    bconv3 = tf.get_variable("bconv3", shape = [256])
    # (6 - 3) / 1 + 1 = 4
    
    # define out graph
    
    # convolutional layer
    a1 = tf.nn.conv2d(X, Wconv1, strides = [1, 1, 1, 1], padding = 'VALID') + bconv1
    # relu activation layer
    h1 = tf.nn.relu(a1)
    # spatial batch normalization
    bn1 = tf.layers.batch_normalization(h1, axis = 1)
    # max pooling
    mp1 = tf.nn.max_pool(value = bn1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # (30 - 2) / 2 + 1 = 15
    
    # convolutional layer
    a2 = tf.nn.conv2d(mp1, Wconv2, strides = [1, 1, 1, 1], padding = 'VALID') + bconv2
    # relu activation layer
    h2 = tf.nn.relu(a2)
    # spatial batch normalization
    bn2 = tf.layers.batch_normalization(h2, axis = 1)
    # max pooling
    mp2 = tf.nn.max_pool(value = bn2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # (13 - 2) / 2 + 1 = 6
    
    # convolutional layer
    a3 = tf.nn.conv2d(mp2, Wconv3, strides = [1, 1, 1, 1], padding = 'VALID') + bconv3
    # relu activation layer
    h3 = tf.nn.relu(a3)
    # spatial batch normalization
    bn3 = tf.layers.batch_normalization(h3, axis = 1)
    # max pooling
    mp3 = tf.nn.max_pool(value = bn3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # (4 - 2) / 2 + 1 = 2
    
    W1 = tf.get_variable("W1", shape = [1024, 1024])
    b1 = tf.get_variable("b1", shape = [1024])
    
    W2 = tf.get_variable("W2", shape = [1024, 1024])
    b2 = tf.get_variable("b2", shape = [1024])
    
    W3 = tf.get_variable("W3", shape = [1024, 1024])
    b3 = tf.get_variable("b3", shape = [1024])
    
    # affine layer
    mp3_flat = tf.reshape(mp3, [-1, 1024])
    a4 = tf.matmul(mp3_flat, W1) + b1
    # relu activation layer
    h4 = tf.nn.relu(a4)
    # affine layer
    a5 = tf.matmul(h4, W2) + b2
    # relu activation layer
    h5 = tf.nn.relu(a5)
    # affine layer
    y_out = tf.matmul(h5, W3) + b3

    return y_out

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X, y, is_training)
mean_loss = None
optimizer = None

# my model : define loss
total_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 1024), logits = y_out) 
mean_loss = tf.reduce_mean(total_loss)

# my model : define optimizer
optimizer = tf.train.RMSPropOptimizer(5e-4)

# batch normalization requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

# train model for 3 epochs then print the validation set accuracy
sess = tf.Session()

sess.run(tf.global_variables_initializer())

print('Training')
run_model(sess, y_out, mean_loss, X_train, y_train, 3, 64, 100, train_step, True)
print('Validation')
run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)   









