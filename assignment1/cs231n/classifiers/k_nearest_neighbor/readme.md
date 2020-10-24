
# KNN
## What's

### What is KNN?


## Why's
### Why should you use KNN?
Though possible, Rule-based approach for image classification is brittle, and scales poorly with new data.

A better approach is Data-driven approach, and KNN is the most simplest data-driven approach.

## How's
### How does `train` work?
Simply memorise train and test data.

### How does `predict` work?
write out the pseudo algorithm here.

### How do you tune the hyper parameter `k`?
By trial & error. Link to cross validation. 

There could be three ways of doing this.
- tune on the entire dataset -> terrible
- split to train / test and tune on the test set -> terrible
- split into  train / validation / test and tune on the validation set -> recommended
- use cross validation -> the gold standard, but beware that it is computationally expensive for dataset with huge size. 

 
## Limits
### Computationally expensive `predict`
In knn, `train` runs in O(1) time (since it is simply "memorizing" the training data), whereas `predict`
 runs in O(n) time, where n is the size of the training data points. 

Generally, what we want is the opposite; we want to spend as much time as possible on `train`
 for the sake of fast `predict`, so that - e.g. we can run the classifier on mobile phones.
 
### Curse of dimensionality

curse of dimensionality |
--- |
![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6d8d9446-70a0-44d2-8644-905adfa646a8/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20201020%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20201020T071838Z&X-Amz-Expires=86400&X-Amz-Signature=deca1aefa1741fde6453ba832deb7777696641e3a2d26c2e85e5d24b52ce25c3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)| 

In principle, in order for KNN to perform at its best, the data space must be densely filled with training data.
Yet, as the dimensionality grows higher (i.e. number of features to train from increases), the number of data points
required to densely fill the data space **grows exponentially**.


## Relations to other algorithms.
For its limits as explained above, KNN is almost never used for image classification 
(though it's a good starting point to learn data-driven algorithms, hyper paramter tuning, etc for its simplicity).

 -> link to linear classifier, svm, neural network, etc.


## Some examples for `numpy` operations

### `np.bincount()`
... fill this in later.

### partial vectorisation - an example
```
A = np.asarray([1])
B = np.asarray([1, 2])
A - B
array([ 0, -1])
B - A
array([0, 1])
```

### `np.sum()` on matrix - how to use `axis` parameter
```
mat
array([[1, 2],
       [3, 4]])
np.sum(mat)
10
np.sum(mat, axis=0)
array([4, 6])
np.sum(mat, axis=1)
array([3, 7])
```


###  broadcast sums

```
m1 = np.array([[1,2], [1,2], [1,2]])
m2 = np.array([[1], [1], [1]])
m1
array([[1, 2],
       [1, 2],
       [1, 2]])
m2
array([[1],
       [1],
       [1]])
m1 + m2
array([[2, 3],
       [2, 3],
       [2, 3]])
m2.shape
(3, 1)
m3 = np.array([[1, 2]])
m3
array([[1, 2]])
m1 + m3
array([[2, 4],
       [2, 4],
       [2, 4]])
```

### what is `xrange` used for?
