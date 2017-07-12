# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:55:15 2017

@author: Haoran You

function: Image Classification by using KNN
 
"""
import numpy as np
import matplotlib.pyplot as plt

from load_data import X_train, Y_train, X_test, Y_test

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

# Visualize some examples from the dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
plt.figure()
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(Y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
        
plt.show()


# reshape the image into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print('reshape the training image feature as:')
print(X_train.shape, X_test.shape)
num_test = Y_test.shape[0]

"""
KNN --- image classifier

"""
from KNN import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, Y_train)
y_pred = classifier.predict(X_test, k = 5, num_loops = 0)
num_correct = np.sum(y_pred == Y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %(num_correct, num_test, accuracy))

"""
# compare the speed of no one and two loop~
def time_function(f, *args):

    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)
"""

# Cross Validation
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array_split(X_train, num_folds)
Y_train_folds = np.array_split(Y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    for fold in range(len(Y_train_folds)):
        fold_train_X = X_train_folds[:]
        fold_train_Y = Y_train_folds[:]
        
        fold_test_X = np.array(fold_train_X.pop(fold))
        fold_test_Y = np.array(fold_train_Y.pop(fold))
        
        fold_train_X = np.vstack(fold_train_X)
        fold_train_Y = np.hstack(fold_train_Y)
        
        classifier.train(fold_train_X, fold_train_Y)
        y_pred = classifier.predict(fold_test_X, k = k, num_loops = 0)
        num_correct = np.sum(y_pred == fold_test_Y)
        accuracy = float(num_correct) / len(fold_test_Y)
        k_to_accuracies[k].append(accuracy)
        
# print out the computed accuracies
for k in sorted(k_to_accuracies):
    a = 0
    for accuracy in k_to_accuracies[k]:
        a += accuracy
    print('k = %d, 5-fold-cross-validation accuracy = %f' % (k, a / num_folds))

    
# plot the raw observations
plt.figure()

for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
    