import numpy as np
import copy

class NeuralNetwork:

    def __init__(self, optimizer_value, weight_initiazer, bias_initiazer):
        self.optimizer = optimizer_value  # Set the optimizer (e.g., SGD, Adam)
        self.loss = []  # Keep track of loss values during training
        self.layers = []  # Layers of the neural network
        self.data_layer = None  # Data input layer
        self.loss_layer = None  # Loss calculation layer
        self.initializers = weight_initiazer, bias_initiazer  # Initializers for weights and biases

    def forward(self):
        # Get data and labels from the data layer
        input_tensor, self.label_tensor = self.data_layer.next()
        # Pass the input through all layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        # Calculate the final prediction using the loss layer
        self.prediction = self.loss_layer.forward(input_tensor, self.label_tensor)
        return self.prediction  # Return the network's prediction

    def backward(self):
        # Compute the loss gradient w.r.t. labels
        loss_ = self.loss_layer.backward(self.label_tensor)
        # Propagate the gradient backward through all layers
        for layer in self.layers[::-1]:
            loss_ = layer.backward(loss_)

    def append_layer(self, layer):
        # If the layer is trainable, initialize its weights and set its optimizer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)  # Copy the optimizer to the layer
            layer.initialize(*self.initializers)  # Initialize weights and biases
        # Add the layer to the list of layers
        self.layers.append(layer)

    def train(self, iterations: int):
        # Train the network for a certain number of iterations
        for _ in range(iterations):
            loss_ = self.forward()  # Get the prediction and loss
            self.backward()  # Perform backpropagation
            self.loss.append(loss_)  # Save the loss for this iteration

    def test(self, input_tensor: np.ndarray):
        # Test the network by passing the input through all layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor  # Return the final output after passing through all layers
