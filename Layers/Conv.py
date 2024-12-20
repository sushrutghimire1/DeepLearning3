from Layers.Base import BaseLayer
import numpy as np

class Conv(BaseLayer):
    def __init__(self, stride_shape, conv_shape, num_kernels):
        # Initialize the convolutional layer
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.conv_shape = conv_shape
        # Determine if the convolution is 1D or 2D
        self.is_1D = False if len(conv_shape) >= 3 else True
        self.is_1x1 = True if conv_shape[1:] == (1, 1) else False
        self.num_kernels = num_kernels
        # Initialize the filters and biases randomly
        self.filters = np.random.uniform(0, 1, size=(num_kernels, *conv_shape))
        self.weights = self.filters
        self.bias = np.random.uniform(0, 1, size=(num_kernels, 1))
        self.input_tensor = None
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    # Property for getting the optimizer
    @property
    def optimizer(self):
        return self._optimizer

    # Property setter for setting the optimizer
    @optimizer.setter
    def optimizer(self, optimizer_value):
        self._optimizer = optimizer_value

    # Property for getting the gradients of the weights
    @property
    def gradient_weights(self):
        return self._gradient_weights

    # Property setter for setting the gradients of the weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    # Property for getting the gradients of the bias
    @property
    def gradient_bias(self):
        return self._gradient_bias

    # Property setter for setting the gradients of the bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    # Padding function to ensure input dimensions match the convolution kernel
    def apply_padding(self, x, pad_before, pad_after):
        if not self.is_1D:
            # Apply padding to a 2D input tensor
            return np.pad(x, ((0, 0), (0, 0), (pad_before[0], pad_after[0]), (pad_before[1], pad_after[1])), 'constant', constant_values=0)
        else:
            # Apply padding to a 1D input tensor
            return np.pad(x, ((0, 0), (0, 0), (pad_before[0], pad_after[0])), 'constant', constant_values=0)

    # Forward pass function for the convolution operation
    def forward(self, input_tensor):
        self.filters = self.weights

        # Extract kernel sizes and compute padding
        kernel_sizes = np.array(self.conv_shape[1:])
        pad_before = np.floor(kernel_sizes / 2).astype(int)
        pad_after = kernel_sizes - pad_before - 1

        m, n_C_prev, n_Y_prev = input_tensor.shape[0:3]
        stride_y = self.stride_shape[0]
        n_C, _, fy = self.filters.shape[0:3]
        # Compute the output size after convolution
        n_Y = (n_Y_prev + pad_before[0] + pad_after[0] - fy) // stride_y + 1


        # Initialize output array based on 1D or 2D
        if self.is_1D:
            output = np.zeros((m, n_C, n_Y))
        else:
            n_X_prev = input_tensor.shape[-1]
            fx = self.filters.shape[-1]
            stride_x = self.stride_shape[1]
            n_X = (n_X_prev + pad_before[1] + pad_after[1] - fx) // stride_x + 1
            output = np.zeros((m, n_C, n_Y, n_X))

        # Pad the input tensor
        padded_input = self.apply_padding(input_tensor, pad_before, pad_after)

        # Perform the convolution operation
        for c in range(n_C):
            filter_ = self.filters[c]
            b = self.bias[c]
            if not self.is_1D:
                # 2D convolution
                for y in range(n_Y):
                    y_start = y * stride_y
                    y_end = y_start + fy
                    for x in range(n_X):
                        x_start = x * stride_x
                        x_end = x_start + fx
                        input_slice = padded_input[:, :, y_start:y_end, x_start:x_end]
                        output[:, c, y, x] = np.sum(input_slice * filter_, axis=(1, 2, 3)) + b
            else:
                # 1D convolution
                for y in range(n_Y):
                    y_start = y * stride_y
                    y_end = y_start + fy
                    input_slice = padded_input[:, :, y_start:y_end]
                    output[:, c, y] = np.sum(input_slice * filter_, axis=(1, 2)) + b

        self.input_tensor = input_tensor
        return output

    # Backward pass function for the gradient computation
    def backward(self, error_tensor):
        kernel_sizes = np.array(self.conv_shape[1:])
        pad_before = np.floor(kernel_sizes / 2).astype(int)
        pad_after = kernel_sizes - pad_before - 1

        if self.is_1D:
            (m, n_C_prev, n_Y_prev) = self.input_tensor.shape
            (n_C, n_C_prev, fy) = self.weights.shape
            stride_y = self.stride_shape[0]
            (m, n_C, n_Y) = error_tensor.shape

            # Initialize gradients
            der_input = np.zeros((m, n_C_prev, n_Y_prev))
            der_weights = np.zeros((n_C, n_C_prev, fy))
            der_bias = np.zeros((n_C, 1))
        else:
            (m, n_C_prev, n_Y_prev, n_X_prev) = self.input_tensor.shape
            (n_C, n_C_prev, fy, fx) = self.weights.shape
            stride_y, stride_x = self.stride_shape
            (m, n_C, n_Y, n_X) = error_tensor.shape

            # Initialize gradients
            der_input = np.zeros((m, n_C_prev, n_Y_prev, n_X_prev))
            der_weights = np.zeros((n_C, n_C_prev, fy, fx))
            der_bias = np.zeros((n_C, 1))

        input_padded = self.apply_padding(self.input_tensor, pad_before, pad_after)
        der_input_padded = self.apply_padding(der_input, pad_before, pad_after)

        # Debugging: Print shapes
        print(f"Input shape: {self.input_tensor.shape}")
        print(f"Padded input shape: {input_padded.shape}")
        print(f"Error tensor shape: {error_tensor.shape}")
        print(f"Kernel size: {self.weights.shape}")
        print(f"Stride: {self.stride_shape}")
        print(f"Pad before: {pad_before}, Pad after: {pad_after}")

        for c in range(n_C):
            filter_ = self.filters[c]
            b = self.bias[c]

            if not self.is_1D:
                # 2D backpropagation
                for y in range(n_Y):
                    y_start = y * stride_y
                    y_end = y_start + fy
                    for x in range(n_X):
                        x_start = x * stride_x
                        x_end = x_start + fx
                        input_slice = input_padded[:, :, y_start:y_end, x_start:x_end]
                        der_input_padded[:, :, y_start:y_end, x_start:x_end] += (
                            filter_ * error_tensor[:, c, y, x][:, None, None, None]
                        )
                        der_weights[c] += np.sum(
                            input_slice * error_tensor[:, c, y, x][:, None, None, None], axis=0
                        )
            else:
                # 1D backpropagation
                for y in range(n_Y):
                    y_start = y * stride_y
                    y_end = y_start + fy
                    input_slice = input_padded[:, :, y_start:y_end]
                    der_input_padded[:, :, y_start:y_end] += (
                        filter_ * error_tensor[:, c, y][:, None, None]
                    )
                    der_weights[c] += np.sum(
                        input_slice * error_tensor[:, c, y][:, None, None], axis=0
                    )

        # Debugging: Check intermediate gradients
        print(f"Gradient weights after accumulation:\n{der_weights}")
        print(f"Gradient bias before accumulation:\n{der_bias}")

        # Accumulate bias gradient for this filter
        if not self.is_1D:
            der_bias = np.sum(error_tensor, axis=(0, 2, 3)).reshape(n_C, 1)
        else:
            der_bias = np.sum(error_tensor, axis=(0, 2)).reshape(n_C, 1)

        print(f"Gradient bias after accumulation:\n{der_bias}")

        # Adjust input gradients based on padding
        if not self.is_1x1:
            if not self.is_1D:
                der_input = der_input_padded[:, :, pad_before[0]:-pad_after[0], pad_before[1]:-pad_after[1]]
            else:
                der_input = der_input_padded[:, :, pad_before[0]:-pad_after[0]]
        else:
            if not self.is_1D:
                der_input = der_input_padded
            else:
                der_input = der_input_padded

        print(f"Gradient input shape (unpadded): {der_input.shape}")

        self.gradient_weights = der_weights
        self.gradient_bias = der_bias

        # Debugging: Final gradients
        print(f"Final gradient weights:\n{self.gradient_weights}")
        print(f"Final gradient bias:\n{self.gradient_bias}")

        # Update weights using the optimizer
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return der_input


    # Initialize weights and biases using the provided initializers
    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            (self.num_kernels, *self.conv_shape), np.prod(self.conv_shape), np.prod(self.conv_shape[1:]) * self.num_kernels
        )
        self.bias = bias_initializer.initialize((self.num_kernels, 1), self.num_kernels, 1)
