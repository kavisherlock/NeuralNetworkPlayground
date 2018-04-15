# Standard libraryimport random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# Activation functions for neurons
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "mneilsonDeepNetwork.py to set\nthe GPU flag to True."


class ConvPoolLayer(object):
    def __init__(self, image_shape, filter_shape, pool_size=(2, 2), activationfn=sigmoid):
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.activationfn = activationfn
        self.layer_input = None
        self.output = None
        self.output_dropout = None

        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
        self.w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                                          dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, layer_input, input_dropout, mini_batch_size):
        self.layer_input = layer_input.reshape(self.image_shape)
        conv_out = conv2d(input=self.layer_input, filters=self.w, filter_shape=self.filter_shape, input_shape=self.image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ws=self.pool_size, ignore_border=True)
        self.output = self.activationfn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activationfn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activationfn = activationfn
        self.p_dropout = p_dropout
        self.layer_input = None
        self.output = None
        self.y_out = None
        self.input_dropout = None
        self.output_dropout = None

        self.w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, layer_input, input_dropout, mini_batch_size):
        self.layer_input = layer_input.reshape((mini_batch_size, self.n_in))
        self.output = self.activationfn((1 - self.p_dropout) * T.dot(self.layer_input, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activationfn(T.dot(self.input_dropout, self.w) + self.b)

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.layer_input = None
        self.output = None
        self.y_out = None
        self.input_dropout = None
        self.output_dropout = None

        self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, layer_input, input_dropout, mini_batch_size):
        self.layer_input = layer_input.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * T.dot(self.layer_input, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(input_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.input_dropout, self.w) + self.b)

    def cost(self, labels):
        return -T.mean(T.log(self.output_dropout)[T.arange(labels.shape[0]), labels])

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.network_input = T.matrix("X")
        self.labels = T.ivector("y")

        initial_layer = self.layers[0]
        initial_layer.set_input(self.network_input, self.network_input, mini_batch_size=self.mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, test_data, epochs, mini_batch_size, eta=0.05, lmbda=0.0):
        training_x, training_y = training_data
        test_x, test_y = test_data

        n_trainingbatches = size(training_data) / mini_batch_size
        n_testbatches = size(test_data) / mini_batch_size

        l2norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self.labels) + 0.5 * lmbda * l2norm_squared / n_trainingbatches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        i = T.lscalar()
        train_mini_batch = theano.function(
            [i],
            cost,
            updates=updates,
            givens={
                self.network_input: training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.labels: training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
            }
        )
        test_accuracy = theano.function(
            [i],
            self.layers[-1].accuracy(self.labels),
            givens={
                self.network_input: test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.labels: test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        cost_ij = 0
        for epoch in xrange(epochs):
            for mini_batch_index in xrange(n_trainingbatches):
                cost_ij = train_mini_batch(mini_batch_index)
            print "Epoch", epoch, ":", cost_ij
            if (epoch + 1) % 10 == 0:
                test_accuracy_value = np.mean([test_accuracy(j) for j in xrange(n_testbatches)])
                print("Test accuracy of {0:.2%}".format(test_accuracy_value))


def size(data):
    return data[0].get_value(borrow=True).shape[0]