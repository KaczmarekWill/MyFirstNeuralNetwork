# Local imports
import mnist_loader 
import NeuralNetwork

# Third-party libraries
import cupy

memory_pool = cupy.cuda.MemoryPool()
cupy.cuda.set_allocator(memory_pool.malloc)
pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

print("")
print("A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network")
print("")
print("This module uses the MNIST digit dataset to construct a model for predicting values of handwritten digits. Modify the hyper-parameters below (or use defaults 1,30,30,10,3) to see how the learning process is affected")
print("")

# Load training data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Develop the network
num_layers = int(input("Number of hidden layers : "))
layers = []
for i in range(0,num_layers):
	layers += [int(input("Neurons on hidden layer : "))]

net = NeuralNetwork.Network([784] + layers + [10])

# Determine the hyper-parameters of the learning algorithm
epochs = int(input("Number of training epochs : "))
mini_batch_size = int(input("Size of mini sample batch : "))
learning_rate = float(input("Learning rate : "))

print("Learning...")

net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

