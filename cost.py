import numpy as np

class MSE:
	'''
	Mean Squarred Error loss Function.
	Used mostly for regression problems.
	'''
	def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
		'''
		Calculates the mean squared error between predictions and targets.

		Args:
			predictions: Array of predictions given by the neural network
			targets: Correct labels for predictions

		Returns:
			calcuated loss
		'''
		return np.mean(np.power(predictions - targets, 2))

	def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
		'''
		Calculates the gradient of the MSE with respect to the predictions.
		'''
		num_samples = len(predictions)

		return 2 * (predictions - targets) / num_samples