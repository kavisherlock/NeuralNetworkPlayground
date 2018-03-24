import numpy as np
import theano
from theano import tensor as T


class Layer(object):
    def __init__(self, layer_input, n_in, n_out, mini_batch_size, activation_fn=T.nnet.sigmoid):
        self.w = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)), dtype=theano.config.floatX))
        self.b = theano.shared(np.asarray(np.zeros((n_out,), dtype=theano.config.floatX)))

        self.input = layer_input.reshape((mini_batch_size, n_in))
        self.output = activation_fn(T.dot(self.input, self.w) + self.b)
        self.params = [self.w, self.b]
        self.y_pred = T.argmax(T.nnet.softmax(self.output), axis=1)

    def cost(self, net):
        return -T.mean(T.log(T.nnet.softmax((T.dot(self.input, self.w) + self.b)))[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_pred))


class Network(object):
    def __init__(self, layer_sizes, mini_batch_size):
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        self.layers = []
        next_input = self.x
        for x, y in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer = Layer(next_input, x, y, mini_batch_size)
            self.layers.append(layer)
            next_input = layer.output

        self.params = [param for layer in self.layers for param in layer.params]

    def SGD(self, training_data, test_data, epochs, mini_batch_size, eta=0.05):
        training_x, training_y = training_data
        test_x, test_y = test_data

        cost = self.layers[-1].cost(self)
        gradients = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, gradients)]
        accuracy = self.layers[-1].accuracy(self.y)

        index = T.lscalar()
        train = theano.function(
            [index], cost, updates=updates,
            givens={
                self.x: training_x[index * mini_batch_size: (index + 1) * mini_batch_size],
                self.y: training_y[index * mini_batch_size: (index + 1) * mini_batch_size]
            }
        )

        predict = theano.function(
            [index], accuracy,
            givens={
                self.x: test_x[index * mini_batch_size: (index + 1) * mini_batch_size],
                self.y: test_y[index * mini_batch_size: (index + 1) * mini_batch_size]
            }
        )

        num_training_batches = training_x.get_value(borrow=True).shape[0] / mini_batch_size
        num_test_batches = test_x.get_value(borrow=True).shape[0] / mini_batch_size

        for epoch in xrange(epochs):
            print epoch
            for i in xrange(num_training_batches):
                costX = train(i)
            print "Epoch:", epoch, "Cost:", costX

        test_accuracy = np.mean([predict(j) for j in xrange(num_test_batches)])
        print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))

