Various ways to implement Neural Networks (and other machine learning techniques)
All are trainied, evaluated and tested using the MNIST digit recognition dataset.
Current best accuracy was measured by running only once on test data after tuning hyperparameters on validation data
The layer sizes [784, 100, 75, 10] is arbitrary that seemed to work pretty well.

Currenty impletmented:
* A reusable simple deep Neural Network class: a neural network from scratch loosely based on Michael Neilson's book Neural Networks and Deep Learning. Current best accuracy on MNIST: 93.16% using a [784, 100, 75, 10] network, 20 epochs, 0.1 learning rate, regularization parameter of 10, mini batch size of 10 and a RELU activation function
* A reusable Logistic Regression class as my introduction to Theano. Trained and tested on the MNIST data. Current best accuracy: 91.17 % using 20 epochs on a mini batch size 10 and learning rate 0.1
* A reusable simple deep Neural Network class using Theano. Current best accuracy on MNIST: 97.01% using a [784, 100, 75, 10] network, 10 epochs, 0.05 learning rate and mini batch size of 10
* A convolution Neural Network class using TensorFlow for MNIST as my introduction to TensorFlow. Current best accuracy on MNIST: 99.1% with four convolution layers followed by a softmax layer
* A reusable simple deep Neural Network class using TensorFlow. Current best accuracy on MNIST: 97.89% using a [784, 100, 75, 10] network, 50 epochs, varying learning rate, mini batch size of 100 and a RELU activation function
* A reusable convolution deep Neural Network class using Theano loosely based on Michael Neilson's book Neural Networks and Deep Learning. Current best accuracy on MNIST: 99.15% with two convolution-max polling layers, one fully connected layer followed by a softmax layer, 40 epochs, minibatch size of 10, 0.1 learning rate, regularization parameter of 0.1 and a tanh activation function