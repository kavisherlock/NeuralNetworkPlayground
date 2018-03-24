import mnist_data_loader
import numpy as np

# from LogisticRegressor import LogisticRegressor
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# lr = LogisticRegressor(784, 10)
# lr.SGD(training_data, test_data, 10, 20)

from simpleNetwork import Network
training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
net = Network([784, 100, 75, 10])
net.SGD(np.array(training_data), np.array(validation_data), 10, 10, 0.5)  # 88.66%

# from simpleNetwork_Theano import Network
# training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
# net = Network([784, 100, 75, 10], 10)
# net.SGD(training_data, validation_data, 10, 10)  # 97.01%

