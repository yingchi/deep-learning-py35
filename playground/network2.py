# -*- coding: utf-8 -*-
# @Author: yingchipei
# @Date:   2017-09-17 17:08:59
# @Last Modified by:   yingchipei
# @Last Modified time: 2017-09-17 23:21:18

"""
network2.py
----------

An improved version of network.py
It implements the stochastic gradient descent learning algorithm for a 
feedforward neural network.
Gradients are calculated using backpropagation.

Improvements include the addition of the cross-entropy cost function, regularization, and better initialization of network weights.

Note: np.dot() is matrix production

"""

import random
import sys
import numpy as np


### Define the quadratic and cross-entropy cost functions as classes
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with the model output a and desired output y"""
        return 0.4*np.linalg.norm(a-y)*2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer"""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with the model output a and desired output y
        The np.nan_to_num ensures that nan is converted to the correct value (0.0).
        because np.log(0) will produce nan.
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))


    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer. 
        Note that the parameter a is not used by the method. It is included to make 
        the interface consistent with the delta method for other cost classes
        """
        return (a-y)


### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        @sizes: a list contains the # neurons in the respective layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian dist with mean 0 and std 1 over the sqaure root 
        of # weights connecting to the same neuron. 
        Initialize the biases using a Gaussian dist with mean 0 and std 1
        """
        self.biases = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialize the weights using a Gaussian dist with mean 0 and std 1.
        Initialize the biases using a Gaussian dist with mean 0 and std 1.
        This weight and bias initializer uses the same approach as in Ch1.
        """
        self.biases = [np.random.rand(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """ 
        Train the neural network using mini-batch stochastic gradient descent.

        @training_data: a list of tuples (x, y) 
        @lambda: regularization parameter
        @evaluation_data: if provided, either the validation data or the test data
        @return: The method returns a tuple containing 4 lists: 
            1. the (per-epoch) costs on the evaluation data
            2. the accuracies on the evaluation data
            3. the costs on the training data
            4. the accuracies on the training data.  
        Note that the lists are empty if the corresponding flag is not set.
        """

        # early stopping functionality:
        best_accuracy = 1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            
            print("Epoch {0} complete".format(j)) 

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print('Cost on trianing data : {}'.format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print('Accuracy on training data: {} / {}'.format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print('Cost on evaluation data: {}'.format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print('Accuracy on evaluation data: {} / {}'
                    .format(self.accuracy(evaluation_data), n_data))

            # early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                else:
                    no_accuracy_change += 1
                if (no_accuracy_change == early_stopping_n):
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying gradient descent using
        backpropagation to a single mini batch. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda/n)) * w - (eta/len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

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
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Return # inputs in `data` for which the neural network outputs the correct result
        @convert: should be False if the data set is validation or test data. True if the 
            data set if the training data. 
            The program usually evaluates the cost on the training data and the accuracy on other 
            data sets. These are different types of computations, and using different 
            representations speeds things up.  More details on the representations can be found in 
            mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """
        Return the total cost for the data set `datq`.
        @convert: should be False if the data set is the training data. True if is the test
            or validation data
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        """
        # no.argmax: Returns the indices of the maximum values along an axis
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


### Miscellaneous functions:
def load(filename):
    """Load a neural network from the file ``filename``.  

    @return: an instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))











