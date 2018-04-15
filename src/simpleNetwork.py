import numpy as np
import random


class Sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))


class Relu(object):
    @staticmethod
    def fn(z):
        return np.maximum(0, z)

    @staticmethod
    def prime(z):
        return np.reshape([1.0 if x > 0 else 0.0 for x in z], np.shape(z))


class Softmax(object):
    @staticmethod
    def fn(z):
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()

    @staticmethod
    def prime(z):
        return Softmax.fn(z) * (1 - Softmax.fn(z))



class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * Sigmoid.prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class Network(object):
    def __init__(self, layer_sizes, activationfn=Sigmoid, cost=CrossEntropyCost):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(x, y)/np.sqrt(x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.bias = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.cost = cost
        self.activationfn = activationfn

    def feed_forward(self, network_input):
        a_values = [network_input]
        z_values = []
        a = network_input
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w.transpose(), a) + b
            z_values.append(z)
            a = self.activationfn.fn(z)
            a_values.append(a)
        network_output = Softmax.fn(z_values[-1])
        return network_output, a_values, z_values

    def backpropogation(self, network_output, a_values, z_values, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.bias]

        delta_cost = self.cost.delta(z_values[-1], network_output, y)
        delta = delta_cost * Softmax.prime(z_values[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(a_values[-2], delta.transpose())

        for l in xrange(2, len(self.layer_sizes)):
            delta = np.dot(self.weights[-l + 1], delta) * self.activationfn.prime(z_values[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(a_values[-l - 1], delta.transpose())

        return grad_w, grad_b

    def SGD(self, training_data, test_data, mini_batch_size, n_epochs=10, eta=0.05, lmbda=0.0, print_train=True):
        for e in xrange(n_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                grad_b = [np.zeros(b.shape) for b in self.bias]
                grad_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    network_output, activations, zvalues = self.feed_forward(x)
                    delta_grad_w, delta_grad_b = self.backpropogation(network_output, activations, zvalues, y)
                    grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
                    grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]

                #  Update the weights and biases
                self.weights = [(1 - eta * (lmbda / len(training_data))) * w - (eta / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, grad_w)]
                self.bias = [b - (eta / len(mini_batch)) * nb
                             for b, nb in zip(self.bias, grad_b)]

            if print_train:
                print 'Epoch', e
                print 'accuracy:', sum(self.n_correct_predictions(mini_batch, True) for mini_batch in mini_batches) * 100. / len(training_data)
                print 'cost:', self.total_cost(training_data, lmbda)
            print self.n_correct_predictions(test_data, True) * 100. / len(test_data)

    def n_correct_predictions(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feed_forward(x)[0]), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)[0]), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda):
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)[0]
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
