# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import cupy as cp

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Based on ``load_data``, but the format is more convenient 
	for use in our implementation of neural networks."""
    tr_d, va_d, te_d = load_data()

	# Reshape the input element of each training image into a 784-dimensional vector
    training_inputs = [cp.reshape(x, (784, 1)) for x in tr_d[0]]
	# Reshape the label element of each training image into a 10-dimensional 
	# vector with the correct label set to 1
    training_results = [vectorized_result(y) for y in tr_d[1]]
	# Zip inputs and results in to 2-tuples
    training_data = zip(training_inputs, training_results)

	# Reshape validation inputs in the same way as training inputs
    validation_inputs = [cp.reshape(x, (784, 1)) for x in va_d[0]]
	# Zip inputs and results in to 2-tuples. Note labels are still integers
    validation_data = zip(validation_inputs, va_d[1])

	# Reshape test inputs in the same way as training images
    test_inputs = [cp.reshape(x, (784, 1)) for x in te_d[0]]
	# Zip inputs and results in to 2-tuples. Again, labels are still integers
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    into a corresponding desired output from the neural network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
