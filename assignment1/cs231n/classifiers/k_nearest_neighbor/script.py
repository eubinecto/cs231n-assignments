from builtins import range
from builtins import object
from typing import Optional

import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# from past.builtins import xrange  # < -- what is xrange for?


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        # to be initialised.
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        # Choose different ways of computing the L2 distance
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X: np.ndarray):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        runs in O(num_test * num_train) time.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data. (a flattened image data)

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        logger = logging.getLogger("compute_distances_two_loops")
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        # initialise the output as zero matrix.
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                # write out the simple solution here (using np.linalg.norm())
                # get the test & train img (1D vector)
                test_flat_img: np.ndarray = X[i]
                train_flat_img: np.ndarray = self.X_train[j]
                # check if they have the same shapes
                assert test_flat_img.shape == train_flat_img.shape
                # compute their L2 distance, only using matrix operations.
                diff = test_flat_img - train_flat_img  # (D,) - (D,) -> (D,)
                diff_square = np.square(diff)  # (D,) -> (D,)
                diff_square_sum = np.sum(diff_square)  # (D,) -> (D,)
                dists[i, j] = np.sqrt(diff_square_sum)  # (D,) -> (D,)
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            else:
                if i % 50 == 0:
                    logger.info("computed dist for:[test={},to all {} train samples]"
                                .format(i, num_train))
        else:
            return dists

    def compute_distances_one_loop(self, X: np.ndarray):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        logger = logging.getLogger("compute_distances_one_loop")
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            diff = self.X_train - X[i]  # (num_train, D) - (D,) -> (num_train, D). partial vectorisation.
            diff_square = np.square(diff)  # (num_train, D) -square-> (num_train, D).
            # you could also do: diff_square = diff**2
            diff_square_sum = np.sum(diff_square, axis=1)  # (num_train, D) -sum-> (num_train, D). sum over rows.
            dists[i] = np.sqrt(diff_square_sum)  # (num_train, D) -sqrt-> (num_train, D)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if i % 50 == 0:
                logger.info("computed dist for:[test={},to all {} train samples]"
                            .format(i, num_train))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        # don't need thoe code below
        # num_test = X.shape[0]
        # num_train = self.X_train.shape[0]
        # dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # here, what this question alludes us is to compute L2 analytically.
        # let x = [x1, x2], y = [y1, y2], then
        # L2(x, y) = sqrt((x1 -y1)^2 + (x2 - y2)^2)
        # = sqrt(x1^2 - 2x1 * y1 + y1^2 + x2^2 - 2x2 * y2 + y2^2)
        # = sqrt(-2(x1 * y1 + x2 * y2) + (x1^2 + x2^2) + (y1^2 + y2^2))
        # = sqrt(-2(x dot y) + sum(square(x)) + sum(square(y)).
        dot_products = X @ self.X_train.T  # (num_test, D) @ (D, num_train) -> (num_test, num_train)
        test_squares = np.square(X)  # (num_test, D) -> (num_test, D)
        train_squares = np.square(self.X_train)   # (num_train, D) -> (num_train, D)
        test_squares_sum = np.sum(test_squares,
                                  # sum over rows
                                  axis=1,
                                  # keep the second dimension as 1
                                  keepdims=True)  # (num_test, D) -> (num_test, 1)
        train_squares_sum = np.sum(train_squares,
                                   axis=1, keepdims=True)  # (num_train, D) -> -> (num_train, 1) (sum over rows)
        dists = np.sqrt(
            -2 * dot_products  # (num_test, num_train)
            + test_squares_sum  # (num_test, 1). broadcast addition to the first dim
            + train_squares_sum.T  # (1, num_train). broadcast addition to the second dim
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists: np.ndarray, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):

            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in knn_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists_to_all_train = dists[i]
            # arg sort the distances in ascending order to get indices for nearest neighbours
            nn_indices = np.argsort(dists_to_all_train)
            # get the indices for k-nearest neighbours
            knn_indices = nn_indices[:k]
            # get the labels for the nearest neighbours
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            knn_y = self.y_train[knn_indices]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # knn_y is a list of non-negative integers (ranging from 0-9),
            # so we can use np.bincount() followed by np.argmax() to find the majority.
            # argmax handles the ties by choosing the smaller label.
            # credit: https://stackoverflow.com/a/6252400
            knn_y_bin_count = np.bincount(knn_y)
            majority = np.argmax(knn_y_bin_count)
            y_pred[i] = majority
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred


if __name__ == '__main__':
    pass
