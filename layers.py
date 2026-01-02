import numpy as np
from typing import Union

class Sigmoid:

	def __init__(self):
		self.cache = None

	def forward(self, z: Union[float, np.ndarray]):
		'''
		Applies the sigmoid function as an activation function for a neuron

		Args:
			z: Value obtained from applying neuron weights and biases

		Returns:
			sigmoid function applied to z
		'''
		self.cache = 1 / (1 + np.exp(-z))
		return self.cache

	def derivative(self):
		'''
		Derivative of the sigmoid activation function for back propagation
		'''
		return self.cache * (1 - self.cache)

class Linear:

	def __init__(self):
		self.cache = None

	def forward(self, z: Union[float, np.ndarray]):
		'''
		Applies the linear activation function for a neuron

		Args:
			z: Value obtained from applying neurons and biases

		Returns:
			z
		'''
		self.cache = z
		return z

	def derivative(self):
		'''
		Derivative of the linear activation function for back propagation
		'''
		if isinstance(self.cache, np.ndarray):
			return np.ones_like(self.cache)
		return 1.0

class Dense:
	'''
	Class representing a dense layer in a neural network.
	The layer receives input from each node in the previous layer.
	'''

	def __init__(self, activation: str, input_size: int, output_size: int):
		'''
		Creates instance of a Dense layer

		Args:
			activation: str value corresponding to an activation function for layer neurons
			input_size: The size of the input that will be received by the layer
			output_size: The number of neurons in the layer that will feed output to the next layer

		Raises:
			ValueError: error raised when non valid activation function is passed
		'''
		self.input_cache = None

		# -- SET ACTIVATION FUNCTION FOR LAYER --
		match activation:
			case 'sigmoid':
				self.activation = Sigmoid()
			case 'linear':
				self.activation = Linear()
			case _:
				raise ValueError("Invalid activation function passed to layer constructor.")

		# -- SET WEIGHTS AND BIASES FOR THE LAYER --
		# Xavier Initialization
		limit = np.sqrt(6 / (input_size + output_size))
		self.weights = np.random.uniform(low=-limit, high=limit, size=(input_size, output_size))
		self.bias = np.zeros((1, output_size))

	def forward(self, inputs: np.ndarray):
		'''
		Passes input through the data layer
		'''
		self.input_cache = inputs 

		# -- APPLY WEIGHTS AND BIASES AND APPLY ACTIVATION
		z = inputs @ self.weights + self.bias 
		return self.activation.forward(z)

	def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01):
		'''
		Performs backward propagation to update parameters.

		Args:
			grad_output: The gradient of the loss with respect to this layer's output (dL/dA)
			learning_rate: The step size for updating weights (Basic SGD).

		Returns:
			grad_input: The gradient of the loss with respect to this layer's input.
							This is passed to the previous layer

		'''
		# -- CALCULATE GRADIENT OF THE ACTIVATION (dZ) --
		# Element-wise multiply incoming gradient by the derivative of the activation
		dz = grad_output * self.activation.derivative()

		# -- CALCULATE GRADIENT FOR WEIGHTS (dW) AND BIASES (db)
		dw = self.input_cache.T @ dz

		db = np.sum(dz, axis=0, keepdims=True)

		grad_input = dz @ self.weights.T 

		self.weights -= learning_rate * dw
		self.bias -= learning_rate * db 

		return grad_input