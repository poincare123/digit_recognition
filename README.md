Description
=============
This is an example of digit recognition program using machine learning algorithms.

Usage
========
The datasets of handwritten digits are originally from MNIST database
http://yann.lecun.com/exdb/mnist/

This code uses a simper format of MNIST database which can be found at
http://pjreddie.com/projects/mnist-in-csv/

Currently, my code uses the KNN algorithm to recognize the digits. It uses
the k-fold validation to find the most optimal value of K in the KNN algorithm.
The final error rate of the KNN algorithm with K=5 is 3.12% based on MNIST
database.

I will add more algorithms such as neural network in the future.

Reference
========
1. The Elements of Statistical Learning
   by Trevor Hastie, Robert Tibshirani, Jerome Friedman
