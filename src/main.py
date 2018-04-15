import mnist_data_loader
import numpy as np

# from LogisticRegressor import LogisticRegressor
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# lr = LogisticRegressor(784, 10)
# lr.SGD(training_data, test_data, 10, 20, 0.1)  # 91.17

# from simpleDeepNetwork import Network
# from simpleDeepNetwork import Relu
# training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
# net = Network([784, 100, 75, 10], activationfn=Relu)
# net.SGD(np.array(training_data), np.array(test_data), 10, 20, eta=0.1, lmbda=10)  # 93.16%

# from simpleDeepNetwork_Theano import Network
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# net = Network([784, 100, 75, 10], 10)
# net.SGD(training_data, test_data, 10, 10)  # 97.01%

# from simpleDeepNetwork_TensorFlow import Network
# training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
# net = Network([784, 100, 75, 10])
# net.SGD(training_data, test_data, 100, 50)  # 97.89

from convolutionDeepNetwork_Theano import sigmoid, tanh, ReLU, Network
from convolutionDeepNetwork_Theano import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
mini_batch_size = 10
net = Network([
    ConvPoolLayer((mini_batch_size, 1, 28, 28), (20, 1, 5, 5), activationfn=tanh),
    ConvPoolLayer((mini_batch_size, 20, 12, 12), (40, 20, 5, 5), activationfn=tanh),
    FullyConnectedLayer(n_in=40*4*4, n_out=100, activationfn=tanh),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, test_data, 50, mini_batch_size, 0.1, 0.1)

