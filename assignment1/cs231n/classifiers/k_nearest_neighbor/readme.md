
# KNN
## What's

### What is KNN?


## Why's
### Why should you use KNN?
Though possible, Rule-based approach for image classification is brittle, and scales poorly with new data.

A better approach is Data-driven approach, and KNN is the most simplest algorithm

## How's
### How does `train` work?
Simply memorise train and test data.

### How does `predict` work?
write out the pseudo algorithm here.

### How do you optimise the hyper parameter `k`?

 
## Limits
### computationally expensive `predict`
In knn, `train` runs in O(1) time (since it is simply "memorizing" the training data), whereas `predict`
 runs in O(n) time, where n is the size of the training data points. 

Generally, what we want is the opposite; we want to spend as much time as possible on `train`
 for the sake of fast `predict`, so that - e.g. we can run the classifier on mobile phones.
 
### curse of dimensionality
In principle, in order for KNN to perform at its best, the data space must be densely filled with training data.
Yet, as the dimensionality grows higher (i.e. number of features to train from increases)

## Alternatives
For its limits as as explained above, KNN is almost never used for image classification 
(though it's a good starting point for its simplicity).

 -> link to linear classifier, svm, neural network, etc.

## questions

Why are we doing multiple loops with KNN?