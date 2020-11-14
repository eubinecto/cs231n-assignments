from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on mini batches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (weight for each class. in different dimensions)
    - X: A numpy array of shape (N, D) containing a mini batch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # dW  = the gradients for each weight. all initialised as zeroes.
    dW = np.zeros(W.shape)   # (Dimension, Classes_num) -> (Dimension, Classes_num)

    # compute the loss and the gradient
    num_classes = W.shape[1]  # W (D, Classes_num)
    num_train = X.shape[0]   # (Num_batch, D)
    loss = 0.0  # starting from zero, accumulate the loss
    for train_idx in range(num_train):
        # the scores for each class
        scores = X[train_idx].dot(W)  # (D,) @ (D, Classes_num) -> (1, Classes_num).  # (1, ) @ (1, C) ->  (1,)
        correct_class_score = scores[y[train_idx]]
        for class_idx in range(num_classes):
            if class_idx == y[train_idx]:
                # as for hinge loss, we compute loss for incorrect answers
                continue
            # delta doesn't matter. (lecture)
            # this is the formula
            margin = scores[class_idx] - correct_class_score + 1  # note delta = 1
            if margin > 0:  # condition.
                loss += margin
                # dL/dW  = d[s_j - s_yi + 1]/dW
                #  = d[s_j] - d[s_yi] - i.e. add the derivative for the incorrect class. subtract for correct class
                # derivative for incorrect class: add
                dW[:, class_idx] += + X[train_idx]
                # derivative for correct class: subtract
                dW[:, y[train_idx]] -= X[train_idx]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train  # derivative should be averaged out as well

    # Add regularization to the loss.
    # reg is the hyper parameter
    # L2 regularisation -> using component-wise multiplication
    loss += reg * np.sum(W * W)  # (D, C) component-wise multiplication (D, C)
    dW += reg * 2 * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # as for the places where
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]  # (D, C) -> C
    num_train = X.shape[0]  # (N, D) -> N
    scores = X.dot(W)  # (N D) (D, C) -> (N, C). scores for each class, for each example
    # previously it was correct_class_score, but now we want to compute it all together.
    # we want a matrix of shape (N, 1). - keep the rows, just choose the correct class (column)
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    # use numpy.maximum for complete vectorisation
    margin = np.maximum(0, scores - correct_class_scores + 1)
    # do not consider correct class in loss
    margin[np.arange(num_train), y] = 0  # otherwise, the value would be 1
    # use the average
    loss = margin.sum() / num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Compute gradient
    margin[margin > 0] = 1
    # Sum over the losses
    valid_margin_count = margin.sum(axis=1)
    # Subtract in correct class (-s_y)
    margin[np.arange(num_train), y] -= valid_margin_count
    # (N, D) -> (D, N) @ (N, C) - > (D, C)
    dW = X.T.dot(margin) / num_train
    # Regularization gradient
    dW = dW + reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
