import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import matplotlib.pyplot as plt
import math

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("tfdata", one_hot=True, reshape=False, validation_size=0)

max_learning_rate = 0.003
min_learning_rate = 0.0001

# Define TensorFlow variables and placeholders
LR = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
B1 = tf.Variable(tf.ones([6])/10)

W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 12], stddev=0.1))
B2 = tf.Variable(tf.ones([12])/10)

W3 = tf.Variable(tf.truncated_normal([4, 4, 12, 24], stddev=0.1))
B3 = tf.Variable(tf.ones([24])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * 24, 200], stddev=0.1))
B4 = tf.Variable(tf.ones([200])/10)

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# feed in 1 when testing, 0.75 when training
pkeep = tf.placeholder(tf.float32)

stride = 1  # output is still 28x28
Y1conv = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1 = tf.nn.relu(Y1conv + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

stride = 2  # output is 14x14
Y2conv = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2 = tf.nn.relu(Y2conv + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

stride = 2  # output is 7x7
Y3conv = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3 = tf.nn.relu(Y3conv + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y3reshaped = tf.reshape(Y3, shape=[-1, 7 * 7 * 24])

Y4 = tf.nn.relu(tf.matmul(Y3reshaped, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Here, TensorFlow computes the partial derivatives of the loss function relatively to all the weights and biases
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
train_step = optimizer.minimize(cross_entropy)

decay_speed = 2000.0

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
accuracies = []
for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(500)
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
    train_data = {X: batch_X, Y_: batch_Y, LR: learning_rate, pkeep: 0.75}

    # train
    sess.run(train_step, feed_dict=train_data)

    if i % 10 == 0:
        accuracies.append(sess.run(accuracy, feed_dict=train_data))
        print i

# success
a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
print "Training:", a * 100, "%"

# success on test data
test_data = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
print "Test:", a * 100, "%"

plt.plot(accuracies)
plt.show()
