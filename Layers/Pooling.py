import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        # Setting the stride and pooling sizes
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False  # Pooling layers are not trainable
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  #the input tensor
       
        (m, n_C, n_Y_prev, n_X_prev) = input_tensor.shape

        pY, pX = self.pooling_shape  
        sY, sX = self.stride_shape  

        # the output size
        n_Y = (n_Y_prev - pY) // sY + 1  
        n_X = (n_X_prev - pX) // sX + 1 

        output = np.zeros((m, n_C, n_Y, n_X))  # Output tensor, initialized with zeros

        # Loop over each position in the output tensor 
        for i in range(n_Y):
            for j in range(n_X):
                y_start = i * sY  
                y_end = y_start + pY  
                x_start = j * sX  
                x_end = x_start + pX 
                # Find the max value in the window and assign it to the output tensor
                output[:, :, i, j] = np.max(input_tensor[:, :, y_start:y_end, x_start:x_end], axis=(2, 3))
                # # ERROR WARNING: maybe I should check if the size matches the input tensor
                # if output[:, :, i, j].shape != (m, n_C):  # Could be a mistake, check later
                #    print("Shape mismatch!!!")

        return output  # Done, output shape should be (m, n_C, n_Y, n_X)

    def backward(self, error_tensor):
        
        (m, n_C, n_Y, n_X) = error_tensor.shape
        pY, pX = self.pooling_shape  # Pooling window
        sY, sX = self.stride_shape  # Stride size
        
        der_input = np.zeros_like(self.input_tensor)  # Initialize derivative as zeros

        # Loop through the error tensor (y, x positions)
        for i in range(n_Y):
            for j in range(n_X):
                y_start = i * sY 
                y_end = y_start + pY  
                x_start = j * sX  
                x_end = x_start + pX  
                input_slice = self.input_tensor[:, :, y_start:y_end, x_start:x_end]  # Get the input slice

               # Double-check
                max_mask = (input_slice == np.max(input_slice, axis=(2, 3), keepdims=True))  # Find max positions in slice
      
                der_input[:, :, y_start:y_end, x_start:x_end] += max_mask * (error_tensor[:, :, i, j])[:, :, np.newaxis, np.newaxis]

        return der_input  # Return the gradient with respect to the input tensor
