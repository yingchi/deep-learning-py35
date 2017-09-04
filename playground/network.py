# -*- coding: utf-8 -*-
# @Author: yingchipei
# @Date:   2017-09-04 15:38:11
# @Last Modified by:   yingchipei
# @Last Modified time: 2017-09-04 17:48:33

"""
network.py
----------

A module to implement the stochastic gradient descent learning algorithm for a 
feedforward neural network.
Gradients are calculated using backpropagation.
It is not optimized, and omits many desirable features

"""

import random
import numpy as numpy


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


class Network(object):

    def __init(self, sizes):
        """
        sizes: a list contains the # neurons in the respective layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ 
        Train the neural network using mini-batch stochastic gradient descent.
        training_data: a list of tuples (x, y) 
        test_data: if provided, then the network will be evaluated against the test data 
        after each epoch, and partial progress printed out. This is useful for tracking
        progress, but slows things down substantially.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) 

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using
        backpropagation to a single mini batch. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]



    def backprop(self, x, y):
        """
        Returns a tule (nable_b, nabla_w) representing the gradient for the cost function C_x.
        nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases
        and self.weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]   # list to store all the activations, layer-by-layer
        zs = []             # list to store all the z vectors, layer-by-layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        











