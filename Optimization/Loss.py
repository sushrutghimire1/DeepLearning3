import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None  # To store predictions for use in backward passs

    def forward(self, prediction_tensor, label_tensor):
        epsilon = np.finfo(float).eps  # Small value to avoid sdivision by 0
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1. - epsilon)  # Clipping predictions
        
        self.prediction_tensor = prediction_tensor  # Save prediction for the backward step
        
        # Cross-entropy loss formula
        loss = -np.sum(label_tensor * np.log(prediction_tensor))
        
        return loss  # Returning computed loss

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps  # To avoid division by zero in the backward pass
        predictions = self.prediction_tensor  
        
        # Compute the gradient of the loss with respect to predictions
        error_tensor = -label_tensor / (predictions + epsilon)
        

        return error_tensor  # Returning computed gradient
