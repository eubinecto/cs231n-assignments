import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n.classifiers.k_nearest_neighbor.script import KNearestNeighbor
from config import ROOT_DIR
from os import path
from dataclasses import dataclass

# the path to CIFAR10 data
# make sure you've run get_datasets.sh script
CIFAR10_DIR = path.join(ROOT_DIR, "assignment1/cs231n/datasets/cifar-10-batches-py")


@dataclass
class Dataset:
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array


def load_cifar10_as_dataset() -> Dataset:
    """
    loads CIFAR_10 dataset from local
    :return: a CIFAR10Dataset object
    """
    # had to put the absolute path. how do I get the root directory?
    global CIFAR10_DIR
    X_train, y_train, X_test, y_test = load_CIFAR10(CIFAR10_DIR)
    return Dataset(X_train, y_train, X_test, y_test)


#
# # Visualize some examples from the dataset.
# # We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()
#
#
# # Subsample the data for more efficient code execution in this exercise
# num_training = 5000
# mask = list(range(num_training))
# X_train = X_train[mask]
# y_train = y_train[mask]
#
# num_test = 500
# mask = list(range(num_test))
# X_test = X_test[mask]
# y_test = y_test[mask]
#
# # Reshape the image data into rows
# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print(X_train.shape, X_test.shape)
#
# # Create a kNN classifier instance.
# # Remember that training a kNN classifier is a noop:
# # the Classifier simply remembers the data and does no further processing
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
#
# # --- classification with KNN classifier --- #
#
#
# # Open cs231n/classifiers/k_nearest_neighbor.py and implement
# # compute_distances_two_loops.
#
# # Test your implementation:
# dists = classifier.compute_distances_two_loops(X_test)
# print(dists.shape)
#
#
# # We can visualize the distance matrix: each row is a single test example and
# # its distances to training examples
# plt.imshow(dists, interpolation='none')
# plt.show()
#
# # -- inline question  1 --- #
#
# # Now implement the function predict_labels and run the code below:
# # We use k = 1 (which is Nearest Neighbor).
# y_test_pred = classifier.predict_labels(dists, k=1)
#
# # Compute and print the fraction of correctly predicted examples
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#
#
# #  You should expect to see approximately `27%` accuracy. Now lets try out a larger `k`, say `k = 5`:
#
#
# y_test_pred = classifier.predict_labels(dists, k=5)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#
# # You should expect to see a slightly better performance than with `k = 1`.
#
# # -- inline question 2 --- #
#
#
# # Now lets speed up distance matrix computation by using partial vectorization
# # with one loop. Implement the function compute_distances_one_loop and run the
# # code below:
# dists_one = classifier.compute_distances_one_loop(X_test)
#
# # To ensure that our vectorized implementation is correct, we make sure that it
# # agrees with the naive implementation. There are many ways to decide whether
# # two matrices are similar; one of the simplest is the Frobenius norm. In case
# # you haven't seen it before, the Frobenius norm of two matrices is the square
# # root of the squared sum of differences of all elements; in other words, reshape
# # the matrices into vectors and compute the Euclidean distance between them.
# difference = np.linalg.norm(dists - dists_one, ord='fro')
# print('One loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')
#
#
# # Now implement the fully vectorized version inside compute_distances_no_loops
# # and run the code
# dists_two = classifier.compute_distances_no_loops(X_test)
#
# # check that the distance matrix agrees with the one we computed before:
# difference = np.linalg.norm(dists - dists_two, ord='fro')
# print('No loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')
#
#
# # Let's compare how fast the implementations are
# def time_function(f, *args):
#     """
#     Call a function f with args and return the time (in seconds) that it took to execute.
#     """
#     import time
#     tic = time.time()
#     f(*args)
#     toc = time.time()
#     return toc - tic
#
# two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop_time)
#
# one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
# print('One loop version took %f seconds' % one_loop_time)
#
# no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop_time)
#
# # You should see significantly faster performance with the fully vectorized implementation!
#
# # NOTE: depending on what machine you're using,
# # you might not see a speedup when you go from two loops to one loop,
# # and might even see a slow-down.
#
# # -- cross validation -- #
#
# num_folds = 5
# k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
#
# X_train_folds = []
# y_train_folds = []
# ################################################################################
# # TODO:                                                                        #
# # Split up the training data into folds. After splitting, X_train_folds and    #
# # y_train_folds should each be lists of length num_folds, where                #
# # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# # Hint: Look up the numpy array_split function.                                #
# ################################################################################
# # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
# pass
#
# # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
# # A dictionary holding the accuracies for different values of k that we find
# # when running cross-validation. After running cross-validation,
# # k_to_accuracies[k] should be a list of length num_folds giving the different
# # accuracy values that we found when using that value of k.
# k_to_accuracies = {}
#
#
# ################################################################################
# # TODO:                                                                        #
# # Perform k-fold cross validation to find the best value of k. For each        #
# # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# # where in each case you use all but one of the folds as training data and the #
# # last fold as a validation set. Store the accuracies for all fold and all     #
# # values of k in the k_to_accuracies dictionary.                               #
# ################################################################################
# # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
# pass
#
# # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
# # Print out the computed accuracies
# for k in sorted(k_to_accuracies):
#     for accuracy in k_to_accuracies[k]:
#         print('k = %d, accuracy = %f' % (k, accuracy))
#
# # plot the raw observations
# for k in k_choices:
#     accuracies = k_to_accuracies[k]
#     plt.scatter([k] * len(accuracies), accuracies)
#
# # plot the trend line with error bars that correspond to standard deviation
# accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
# accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
# plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# plt.title('Cross-validation on k')
# plt.xlabel('k')
# plt.ylabel('Cross-validation accuracy')
# plt.show()
#
#
# # Based on the cross-validation results above, choose the best value for k,
# # retrain the classifier using all the training data, and test it on the test
# # data. You should be able to get above 28% accuracy on the test data.
# best_k = 1
#
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
# y_test_pred = classifier.predict(X_test, k=best_k)
#
# # Compute and display the accuracy
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#
#
# # -- inline question 3 -- #

if __name__ == '__main__':
    print("### As a sanity check, we print out the size of the training and test data. ###")
    cifar10 = load_cifar10_as_dataset()
    print('---Training data shape:')
    print(cifar10.X_train.shape)
    print("(num_train, img_width, img_height, channel_size)")
    print('---Training labels shape:')
    print(cifar10.y_train.shape)
    print("(num_train,)")
    print('---Test data shape:')
    print(cifar10.X_test.shape)
    print("(num_test, img_width, img_height, channel_size)")
    print('---Test labels shape:')
    print(cifar10.y_test.shape)
    print("(num_test,)")

