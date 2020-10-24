import random
from typing import Tuple, List, Dict
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from assignment1.cs231n.classifiers.k_nearest_neighbor.script import KNearestNeighbor
from config import ROOT_DIR
from os import path
from dataclasses import dataclass
# for de-denting
# doc: https://docs.python.org/3/library/textwrap.html#textwrap.dedent
import textwrap
import logging
# the path to CIFAR10 data
# make sure you've run get_datasets.sh script
CIFAR10_DIR = path.join(ROOT_DIR, "assignment1/cs231n/datasets/cifar-10-batches-py")
# the 10 classes of CIFAR10 dataset.
# the idx to each class is the value for y_train, y_test
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# the number of sub samples to use for this exercise
# not using the entire dataset for efficient execution (remember, prediction with KNN runs in O(n) time)
NUM_TRAIN = 5000
NUM_TEST = 500
NUM_FOLDS = 5
K_CHOICES = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
RANDOM_SEED = 715
random.seed(RANDOM_SEED)


@dataclass
class Dataset:
    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array


def load_cifar10_as_dataset() -> Dataset:
    # had to put the absolute path. how do I get the root directory?
    global CIFAR10_DIR
    X_train, y_train, X_test, y_test = load_CIFAR10(CIFAR10_DIR)
    return Dataset(X_train, y_train, X_test, y_test)


def visualise_cifar10(cifar10: Dataset):
    global CLASSES
    num_classes = len(CLASSES)
    samples_per_class = 7
    for y, cls in enumerate(CLASSES):
        # boolean operator on matrices -> outputs either 0 or 1
        # flatnonzero: "Return indices that are non-zero in the flattened version of a.
        # This is equivalent to np.nonzero(np.ravel(a))[0]."
        idxs = np.flatnonzero(cifar10.y_train == y)
        # random selection without replacement
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(cifar10.X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def subsample_cifar10(cifar10: Dataset):

    global NUM_TRAIN, NUM_TEST
    # range returns a range object, which is a generator, not an iterable.
    # but it can be used for list slicing. (so no need for list conversion here)
    cifar10.X_train = cifar10.X_train[range(NUM_TRAIN)]
    cifar10.y_train = cifar10.y_train[range(NUM_TRAIN)]
    cifar10.X_test = cifar10.X_test[range(NUM_TEST)]
    cifar10.y_test = cifar10.y_test[range(NUM_TEST)]


def flatten_img_data(cifar10: Dataset):
    # get the number of image samples
    num_train_img = cifar10.X_train.shape[0]
    num_test_img = cifar10.X_test.shape[0]
    # flatten image data into rows
    # * newshape: The new shape should be compatible with the original shape.
    # If an integer, then the result will be a 1-D array of that length.
    # One shape dimension can be -1. In this case, the value is inferred
    # from the length of the array and remaining dimensions.
    cifar10.X_train = np.reshape(cifar10.X_train,
                                 # flatten 4D array to 2D array, where
                                 # first dimension = num_train_img (same as before)
                                 # but the other dimension = infer from the rest of dimensions.
                                 # in this case = 32 * 32 * 3 = 3072.
                                 # (np.shape will infer this for you if you put in -1).
                                 newshape=(num_train_img, -1))
    cifar10.X_test = np.reshape(cifar10.X_test, newshape=(num_test_img, -1))


#  --- classification with KNN classifier --- #

def visualise_dists(dists: np.ndarray):
    plt.imshow(dists, interpolation='none')
    plt.show()

# --- check inline question  1 --- #


def predict_and_eval(knn_cls: KNearestNeighbor, y_test: np.ndarray,
                     dists: np.ndarray, k: int) -> float:
    y_test_pred = knn_cls.predict_labels(dists, k)
    # Compute and print the fraction of correctly predicted examples
    num_correct = np.sum(y_test_pred == y_test)
    num_test = y_test.shape[0]
    accuracy = float(num_correct) / num_test
    print('Got {} / {} correct => accuracy: {}'
          .format(num_correct, num_test, accuracy))
    return accuracy


#  --- check inline question 2 --- #

def frobenius_compare(mat1: np.ndarray, mat2: np.ndarray):
    """
    ################################################################################
    To ensure that our vectorized implementation is correct, we make sure that it
    agrees with the naive implementation. There are many ways to decide whether
    two matrices are similar; one of the simplest is the Frobenius norm. In case
    you haven't seen it before, the Frobenius norm of two matrices is the square
    root of the squared sum of differences of all elements; in other words, reshape
    the matrices into vectors and compute the Euclidean distance between them.
    ################################################################################
    """
    difference = np.linalg.norm(mat1 - mat2, ord='fro')
    print('One loop difference was: %f' % (difference, ))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')


def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


#  -- cross validation -- #
def split_train_to_folds(cifar10: Dataset) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    """
    global NUM_FOLDS
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_train_folds = np.array_split(ary=cifar10.X_train, indices_or_sections=NUM_FOLDS)
    y_train_folds = np.array_split(ary=cifar10.y_train, indices_or_sections=NUM_FOLDS)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return X_train_folds, y_train_folds


def perform_k_fold_cv(knn_cls: KNearestNeighbor,
                      X_train_folds: List[np.ndarray],
                      y_train_folds: List[np.ndarray]):
    """
    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################
    """
    logger = logging.getLogger("perform_k_fold_cv")
    global K_CHOICES, NUM_FOLDS
    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies: Dict[int, list] = {}
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for k in K_CHOICES:
        # collect accuracies for this k
        accs = list()
        logger.info("====== {}-fold CV to compute accs for k={} =======".format(NUM_FOLDS, k))
        for fold_idx in range(NUM_FOLDS):
            logger.info("fold:" + str(fold_idx))
            # train / validation split
            X_valid_batch = X_train_folds.pop(fold_idx)
            y_valid_batch = y_train_folds.pop(fold_idx)
            X_train_batch = np.concatenate(X_train_folds)
            y_train_batch = np.concatenate(y_train_folds)
            # for next iteration
            X_train_folds.insert(fold_idx, X_valid_batch)
            y_train_folds.insert(fold_idx, y_valid_batch)
            # train, compute dists, predict and eval
            knn_cls.train(X=X_train_batch, y=y_train_batch)
            dists = knn_cls.compute_distances_no_loops(X=X_valid_batch)
            acc = predict_and_eval(knn_cls=knn_cls, y_test=y_valid_batch, dists=dists, k=k)
            accs.append(acc)
        else:
            # store the list of accuracies to the dict
            k_to_accuracies[k] = accs
            logger.info("accs for k={} is:{}".format(k, k_to_accuracies[k]))
    else:
        return k_to_accuracies
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def plot_cv_result(k_to_accs: Dict[int, list]):
    # Print out the computed accuracies
    for k in sorted(k_to_accs):
        for accuracy in k_to_accs[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))
    # plot the raw observations
    for k in K_CHOICES:
        accuracies = k_to_accs[k]
        plt.scatter([k] * len(accuracies), accuracies)
    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accs.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accs.items())])
    plt.errorbar(K_CHOICES, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


# -- check  inline question 3 -- #


def main():
    print("======= preprocessing cifar10 dataset ========")
    print(textwrap.dedent(
        """
        #######################################
        load CIFAR_10 dataset from local.
        #######################################
        """
    ))
    cifar10 = load_cifar10_as_dataset()
    print("As a sanity check, we print out the size of the training and test data.")
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
    # visualise
    print(textwrap.dedent(
        """
        #######################################
        Visualize some examples from the dataset.
        We show a few examples of training images from each class.
        #######################################
        """
    ))
    visualise_cifar10(cifar10)
    print("check the plot view.")
    # subsample
    print(textwrap.dedent(
        """
        #######################################
        Subsample the data for more efficient code execution in this exercise
        #######################################
        """
    ))
    subsample_cifar10(cifar10)
    print('---Training data shape:')
    print(cifar10.X_train.shape)
    print('---Test data shape:')
    print(cifar10.X_test.shape)
    # flatten
    print(textwrap.dedent(
        """
        #######################################
        Reshape the image data into rows
        #######################################
        """
    ))
    flatten_img_data(cifar10)
    print('---Training data shape:')
    print(cifar10.X_train.shape)
    print('---Test data shape:')
    print(cifar10.X_test.shape)
    # knn classification
    print("\n======== classifying cifar10 image with KNN classifier =========")
    print(textwrap.dedent(
        """
        #######################################
        Create a kNN classifier instance.
        Remember that training a kNN classifier is a noop:
        the Classifier simply remembers the data and does no further processing
        #######################################
        """
    ))
    # instantiate a knn classifier
    knn_cls = KNearestNeighbor()
    knn_cls.train(X=cifar10.X_train, y=cifar10.y_train)
    print("no change to the training set:")
    print(np.array_equal(knn_cls.X_train, cifar10.X_train))
    # compute dists
    print(textwrap.dedent(
        """
        #######################################
        Open cs231n/classifiers/k_nearest_neighbor.py and implement
        compute_distances_two_loops.
        Test your implementation:
        #######################################
        """
    ))
    dists = knn_cls.compute_distances_two_loops(cifar10.X_test)
    print("--- the shape of the dists matrix")
    print(dists.shape)
    # visualise dists
    print(textwrap.dedent(
        """
        #######################################
        We can visualize the distance matrix: each row is a single test example and
        its distances to training examples
        #######################################
        """
    ))
    visualise_dists(dists)
    print("check the plot view")
    # predict and eval
    print(textwrap.dedent(
        """
        #######################################
        Now implement the function predict_labels and run the code below,
        which predicts the labels for all test examples and evaluates accuracy.
        #######################################
        """
    ))
    print("---when k=1, you should expect to see approximately `27%` accuracy.")
    predict_and_eval(knn_cls, cifar10.y_test, dists, k=1)
    print("---Now lets try out a larger `k`, say `k = 5`. you should expect to see a slightly better performance")
    predict_and_eval(knn_cls, cifar10.y_test, dists, k=5)
    # compute dists again
    print(textwrap.dedent(
        """
        #######################################
        Now lets speed up distance matrix computation by using partial vectorization
        with one loop. Implement the function compute_distances_one_loop and run the
        code below:
        #######################################
        """
    ))
    dists_one = knn_cls.compute_distances_one_loop(cifar10.X_test)
    print(textwrap.dedent(frobenius_compare.__doc__))
    frobenius_compare(dists, dists_one)

    print(textwrap.dedent(
        """
        #######################################
        Now lets speed up distance matrix computation by using complete vectorization
        with no loops. Implement the function compute_distances_no_loop and run the
        code below:
        #######################################
        """
    ))
    dists_no = knn_cls.compute_distances_no_loops(cifar10.X_test)
    print(textwrap.dedent(frobenius_compare.__doc__))
    frobenius_compare(dists, dists_no)

    print(textwrap.dedent(
        """
        #######################################
        Running time comparison of the three algorithms for computing L2 dist.
        You should see significantly faster performance with the fully vectorized implementation!
        NOTE: depending on what machine you're using,
        you might not see a speedup when you go from two loops to one loop,
        and might even see a slow-down.
        #######################################
        """
    ))
    two_loop_time = time_function(knn_cls.compute_distances_two_loops, cifar10.X_test)
    print('Two loop version took %f seconds' % two_loop_time)
    one_loop_time = time_function(knn_cls.compute_distances_one_loop, cifar10.X_test)
    print('One loop version took %f seconds' % one_loop_time)
    no_loop_time = time_function(knn_cls.compute_distances_no_loops, cifar10.X_test)
    print('No loop version took %f seconds' % no_loop_time)
    # split train to folds for cross validation
    print(textwrap.dedent(split_train_to_folds.__doc__))
    X_train_folds, y_train_folds = split_train_to_folds(cifar10)
    print("X_train folds: " + str(X_train_folds))
    print("y_train_folds: " + str(y_train_folds))
    # perform k-fold cv
    print(textwrap.dedent(perform_k_fold_cv.__doc__))
    k_to_accs = perform_k_fold_cv(knn_cls, X_train_folds, y_train_folds)
    print("k_to_accs: " + str(k_to_accs))
    plot_cv_result(k_to_accs)
    print(textwrap.dedent(
        """
        #######################################
        Based on the cross-validation results above, choose the best value for k,
        retrain the classifier using all the training data, and test it on the test
        data. You should be able to get above 28% accuracy on the test data.
        #######################################
        """
    ))
    best_k = 10
    print("best k: " + str(best_k))
    knn_cls.train(cifar10.X_train, cifar10.y_train)
    dists = knn_cls.compute_distances_no_loops(cifar10.X_test)
    predict_and_eval(knn_cls, cifar10.y_test, dists, k=best_k)


if __name__ == '__main__':
    # doing it this way to avoid "shadows xyz from outer scope" warning
    # credit: https://stackoverflow.com/a/31575708
    main()
