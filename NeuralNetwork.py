import random
import numpy as np
import cupy as cp


class Network(object):

    def __init__(self, sizes):
        """The list `sizes` contains the number of neurons in the
        respective layers of the network. The biases and weights for the
        network are initialized randomly."""
        self.num_layers = len(sizes)
        self.sizes = sizes
		# Returns samples from a 'standard normal' distribution. Randomizes biases
		# Creates a list of bias for each layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		# Returns samples from a 'standard normal' distribution. Randomizes weights
		# Creates a vector of weights for each input to the neuron
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network with `a` as input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(cp.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. If `test_data` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out."""
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        `eta` is the learning rate."""
		# Returns arrays of zeroes with the shape of each bias/weight
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
			# Backpropogate, returns changes to biases and weights
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			# Update nabla_b/w after each mini_batch backprops
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		# Adjusts biases/weights with respect to learning speed and
		# mini batch size
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple `(nabla_b, nabla_w)` representing the
        gradient for the cost function C_x.  `nabla_b` and
        `nabla_w` are layer-by-layer lists of cupy arrays, similar
        to `self.biases` and `self.weights`."""
		# Returns arrays of zeroses with the shape of each bias/weight
        nabla_b = [cp.zeros(b.shape) for b in self.biases]
        nabla_w = [cp.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x # mini_batch input
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
			# z is the sum of weights * activations
            z = cp.dot(w, activation)+b
            zs.append(z)
			# sigmoid clamps z to 0-1
            activation = sigmoid(z)
            activations.append(activation)
        # BACKWARD PASS - THIS IS THE PART WHERE IT LEARNS
		# delta is the difference between the final activation and
		# the expected activation, times the slope of the final vector
		# of z
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
		# weight is the dot product of delta and previous layer
		# activations (vector is rotated here)
        nabla_w[-1] = cp.dot(delta, activations[-2].transpose())
		# repeat this process for each layer, moving backwards
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = cp.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = cp.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
		# Sets result to highest activation in final layer
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
		# Returns sum of correct guesses
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

