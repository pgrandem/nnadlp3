"""rep.py 
to use the neural network example since I don't have python environment
"""

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 0, 10])
net.SGD(training_data,30,10,1.0, test_data=test_data)
