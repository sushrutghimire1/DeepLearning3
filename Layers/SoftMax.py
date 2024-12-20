import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None  # Store output probabilities for backpropagation

    def forward(self, input):
       
        # Validate input dimensions
        if len(input.shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {input.shape}")

        # Check for invalid values in the input
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            raise ValueError("Input contains NaN or infinite values.")

        # Stabilize computation by subtracting the maximum value in each row
        shifted_input = input - np.max(input, axis=1, keepdims=True)

        # Compute exponentials and their sum
        exp_values = np.exp(shifted_input)
        exp_sums = np.sum(exp_values, axis=1, keepdims=True)

        # Compute probabilities
        self.output = exp_values / exp_sums
        return self.output

    def backward(self, grad_output):
       
        batch_size, num_classes = self.output.shape
        input_grad = np.zeros_like(grad_output)

        # Loop-based implementation of Jacobian vector product
        for i in range(batch_size):
            softmax_vec = self.output[i].reshape(-1, 1)  # Column vector
            jacobian_matrix = np.diagflat(softmax_vec) - np.dot(softmax_vec, softmax_vec.T)
            input_grad[i] = np.dot(jacobian_matrix, grad_output[i])

        return input_grad

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialization is not needed for SoftMax, but the method is defined for consistency.
        """
        pass
