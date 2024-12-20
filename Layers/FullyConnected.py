import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        # Just initializing stuff
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # Randomly initialize weights between input and output size, adding one extra for the bias
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self.input_tensor = None
        self._gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        # Setting up weights using the initializer provided
        self.weights[:-1, :] = weights_initializer.initialize(
            (self.input_size, self.output_size), self.input_size, self.output_size
        )
        # Bias initialization goes in the last row
        self.weights[-1, :] = bias_initializer.initialize(
            (1, self.output_size), 1, self.output_size
        )

    def forward(self, input_tensor):
        # First, we add a column of ones to the input tensor to handle the bias
        bias_term = np.ones((input_tensor.shape[0], 1))
        self.input_tensor = np.hstack((input_tensor, bias_term))  # Adding the bias column to the input
        # Output is just a dot product between the input and weights
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        # We need to compute the gradients: first for the weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # Then for the input, by multiplying error with the weights (ignoring the bias part)
        gradient_input = np.dot(error_tensor, self.weights[:-1].T)

        # If there's an optimizer, update the weights
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        return gradient_input

    @property
    def optimizer(self):
        # Simple getter for the optimizer
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        # And the setter to assign the optimizer
        self._optimizer = opt

    @property
    def gradient_weights(self):
        # Just a getter for the weight gradients
        return self._gradient_weights
