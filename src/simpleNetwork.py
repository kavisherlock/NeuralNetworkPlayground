import numpy as np
import random


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a-y


class Network(object):
    def __init__(self, layer_sizes, cost=CrossEntropyCost):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(x, y) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.bias = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.cost = cost

    def feed_forward(self, network_input):
        a_values = [network_input]
        z_values = []
        a = network_input
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w.transpose(), a) + b
            z_values.append(z)
            a = sigmoid(z)
            a_values.append(a)
        return a_values, z_values

    def backpropogation(self, a_values, z_values, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.bias]

        delta_cost = self.cost.delta(z_values[-1], a_values[-1], y)
        delta = delta_cost * sigmoid_prime(z_values[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(a_values[-2], delta.transpose())

        for l in range(len(self.layer_sizes)-1, 1, -1):
            delta = np.dot(self.weights[l-1], delta) * sigmoid_prime(z_values[l-2])
            grad_w[l-2] = np.dot(a_values[l-2], delta.transpose())
            grad_b[l-2] = delta

        return grad_w, grad_b

    def SGD(self, training_data, test_data, mini_batch_size, n_epochs, eta=0.05):
        for e in xrange(n_epochs):
            print e
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                grad_b = [np.zeros(b.shape) for b in self.bias]
                grad_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    activations, zvalues = self.feed_forward(x)
                    delta_grad_w, delta_grad_b = self.backpropogation(activations, zvalues, y)
                    grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
                    grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]

                # Update the weights and biases
                self.weights = [w - (eta / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, grad_w)]
                self.bias = [b - (eta / len(mini_batch)) * nb
                             for b, nb in zip(self.bias, grad_b)]

        mini_test_batches = [test_data[k:k + mini_batch_size] for k in xrange(0, len(test_data), mini_batch_size)]
        print sum(self.accuracy(mini_batch) for mini_batch in mini_test_batches) * 100. / len(test_data)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feed_forward(x)[0][-1]), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)[0][-1]), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


def sigmoid(z):
    # The sigmoid function.
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # The derivative of the sigmoid function.
    return sigmoid(z) * (1 - sigmoid(z))
