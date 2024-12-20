import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None
        self.trainable = False  # Indicates that this layer has no trainable parameters

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape  # Save input shape for backpropagation
        batch_size = self.input_shape[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
    
    def initialize(self, weights_initializer, bias_initializer):
        # No initialization needed for pooling
        pass
