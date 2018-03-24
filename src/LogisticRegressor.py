import numpy
import theano
from theano import tensor as T
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'

rng = numpy.random


class LogisticRegressor(object):
    def __init__(self, n_features, n_classes):
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        self.W = theano.shared(rng.randn(n_features, n_classes), name="W")
        self.b = theano.shared(numpy.zeros(n_classes,), name="b")

    def SGD(self, training_data, test_data, mini_batch_size, epochs, eta=0.05):
        training_x, training_y = training_data
        test_x, test_y = test_data

        prediction = T.nnet.softmax((T.dot(self.x, self.W) + self.b))
        accuracy = T.mean(T.eq(self.y, T.argmax(prediction, axis=1)))

        cost = -T.mean(T.log(prediction)[T.arange(self.y.shape[0]), self.y])
        grad_w, grad_b = T.grad(cost, [self.W, self.b])
        updates = ((self.W, self.W - 0.1 * grad_w), (self.b, self.b - 0.1 * grad_b))

        i = T.lscalar()  # mini-batch index
        train = theano.function(
            inputs=[i],
            outputs=[cost],
            updates=updates,
            givens={
                self.x: training_x[i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y: training_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )
        # predict = theano.function(inputs=[self.x], outputs=prediction)
        evaluate = theano.function(
            inputs=[i],
            outputs=accuracy,
            givens={
                self.x: test_x[i * mini_batch_size: (i + 1) * mini_batch_size],
                self.y: test_y[i * mini_batch_size: (i + 1) * mini_batch_size]
            }
        )

        num_training_batches = training_x.get_value(borrow=True).shape[0] / mini_batch_size
        num_test_batches = test_x.get_value(borrow=True).shape[0] / mini_batch_size

        # print("Initial model:")
        # print(self.W.get_value())
        # print(self.b.get_value())

        err = -1
        for epoch in xrange(epochs):
            print epoch
            for i in xrange(num_training_batches):
                err = train(i)

        # print("Training donezo:")
        # print("Final model:")
        # print(self.W.get_value())
        # print(self.b.get_value())

        test_accuracy = numpy.mean([evaluate(j) for j in xrange(num_test_batches)])
        print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))


