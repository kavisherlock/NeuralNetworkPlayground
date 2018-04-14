import math
import random
import numpy as np
import tensorflow as tf


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


class Network(object):
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.n_classes = layer_sizes[-1]
        self.x = tf.placeholder(tf.float32, [None, layer_sizes[0], 1])
        self.pkeep = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.decay_speed = 2000

        self.weights = []
        self.biases = []
        for j, k in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(tf.Variable(tf.truncated_normal([j, k], stddev=0.1)))
            self.biases.append(tf.Variable(tf.ones([k])/10))

        layer_input = tf.reshape(self.x, [-1, layer_sizes[0]])
        for i in range(0, self.num_layers - 2):
            layer_output = tf.nn.relu(tf.matmul(layer_input, self.weights[i]) + self.biases[i])
            layer_output_dropped = tf.nn.dropout(layer_output, self.pkeep)
            layer_input = layer_output_dropped
        ylogits = tf.matmul(layer_input, self.weights[-1]) + self.biases[-1]
        self.y = tf.nn.softmax(ylogits)

        # placeholder for correct labels
        self.y_ = tf.placeholder(tf.float32, [None, self.n_classes])

        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=ylogits, labels=self.y_)
        cross_entropy = tf.reduce_mean(cross_entropy) * 100

        # % of correct answers found in batch
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # Here TensorFlow computes the partial derivatives of the loss function relatively to all the weights and biases
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.minimize(cross_entropy)

    def SGD(self, training_data, test_data, mini_batch_size, n_epochs=10, min_learning_rate=0.0001, max_learning_rate=0.003):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(0, n_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            xrange(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                # load batch of images and correct answers
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / self.decay_speed)
                batch_x = []
                batch_y = []
                for x, y in mini_batch:
                    batch_x.append(x)
                    batch_y.append(y)
                train_data_dict = {self.x: np.asarray(batch_x), self.y_: np.asarray(batch_y).reshape([mini_batch_size, 10]), self.learning_rate: learning_rate, self.pkeep: 0.75}

                # train
                sess.run(self.train_step, feed_dict=train_data_dict)

            if epoch % 10 == 0:
                print 'Epoch', epoch
                print sess.run(self.accuracy, feed_dict=train_data_dict) * 100

        # success on test data
        mini_test_batches = [test_data[k:k + mini_batch_size] for k in xrange(0, len(test_data), mini_batch_size)]
        a = 0
        for mini_batch in mini_test_batches:
            batch_x = []
            batch_y = []
            for x, y in mini_batch:
                batch_x.append(x)
                batch_y.append(y)
            test_data_dict = {self.x: batch_x, self.y_: np.asarray(batch_y).reshape([mini_batch_size, self.n_classes]), self.pkeep: 1}
            a += sess.run(self.accuracy, feed_dict=test_data_dict)
        a = a / len(mini_test_batches)

        print "Test:", a * 100., "%"

