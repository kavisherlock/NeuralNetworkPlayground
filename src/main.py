import mnist_data_loader
import numpy as np

# from LogisticRegressor import LogisticRegressor
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# lr = LogisticRegressor(784, 10)
# lr.SGD(training_data, test_data, 10, 20, 0.1)  # 91.17

from simpleNetwork import Network
from simpleNetwork import Relu
training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
net = Network([784, 100, 75, 10], activationfn=Relu)
net.SGD(np.array(training_data), np.array(test_data), 10, 20, eta=0.1, lmbda=10)  # 93.16%

# from simpleNetwork_Theano import Network
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# net = Network([784, 100, 75, 10], 10)
# net.SGD(training_data, test_data, 10, 10)  # 97.01%

# from simpleNetwork_TensorFlow import Network
# training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
# net = Network([784, 100, 75, 10])
# net.SGD(training_data, test_data, 100, 50)  # 97.89

