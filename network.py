import layers
import cost

import numpy as np


class Network:

	def __init__(self):
		self.layers = []
		self.loss = None

	def add_layer(self, layer):
		'''
		Adds a new layer to the neural network

		Args:
			layer: A layer defined in layers.py
		'''
		self.layers.append(layer)

	def set_loss(self, loss):
		self.loss = loss

	def forward(self, inputs: np.ndarray):
		'''
		Makes a prediction using defined network layers

		Args:
			inputs: Vector of input data to pass to the network

		Returns:
			network prediction
		'''
		for layer in self.layers:
			inputs = layer.forward(inputs)
		return inputs

	def backward(self, grad_output: np.ndarray, learning_rate: float):
		'''
		Passes gradients backward through all layers to update weights.

		Args:
			grad_output: The gradient from the next layer
			learning_rate: Value which dictates the rate of training
		'''
		for layer in reversed(self.layers):
			grad_output = layer.backward(grad_output, learning_rate)

	def train(self, inputs: np.ndarray, targets: np.ndarray, loss_function = None, epochs: int = 1000, learning_rate: float = 0.01):
		'''
		Moves forward through the network.
		Calculates loss.
		Performs backward propogation to update the weights and biases

		Args:
			inputs: Training data
			targets: Training labels
			loss_function: Instance of a Loss class
			epochs: Number of times to loop over the dataset
			learning_rate: Step size for weight updates

		Raises:
			ValueError: Error raised if no loss function given
		'''
		if loss_function is None:
			loss_function = self.loss

		if loss_function is None:
			raise ValueError('No loss function given.')

		for i in range(epochs):
			predictions = self.forward(inputs)
			loss_val = loss_function.forward(predictions, targets)
			loss_grad = loss_function.backward(predictions, targets)
			self.backward(loss_grad, learning_rate)

			if i % 100 == 0:
				print(f"Epoch {i}/{epochs} - Loss: {loss_val:.6f}")

if __name__ == '__main__':
	from layers import Dense 
	from cost import MSE 
	# 1. Setup Data (XOR Problem)
	# The network tries to learn: 
	# [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
	X = np.array([[0,0], [0,1], [1,0], [1,1]])
	y = np.array([[0], [1], [1], [0]])

	# 2. Build Network
	net = Network()

	# Input Layer (2 inputs) -> Hidden Layer (3 neurons, Sigmoid)
	net.add_layer(Dense('sigmoid', input_size=2, output_size=3))

	# Hidden Layer (3 inputs) -> Output Layer (1 neuron, Linear)
	# Note: For binary classification, Sigmoid is usually better here, 
	# but Linear works for this demo if we treat it as regression close to 0/1.
	net.add_layer(Dense('linear', input_size=3, output_size=1))

	# 3. Define Cost Function
	mse = MSE()

	# 4. Train
	print("Starting training...")
	net.train(inputs=X, targets=y, loss_function=mse, epochs=5000, learning_rate=0.1)

	# 5. Test
	print("\nFinal Predictions:")
	predictions = net.forward(X)
	print(predictions)