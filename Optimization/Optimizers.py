import numpy as np

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None  # Initialize velocity to None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)  # Set velocity to zero if it's the first update

        # Update the velocity using momentum and gradient
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor

        # Apply the update to the weights
        weight_update = weight_tensor + self.velocity
        return weight_update  # Return the updated weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu  # beta1, controls momentum
        self.rho = rho  # beta2, controls the second moment estimate
        self.epsilon = 1e-8  # Small value to avoid division by zero
        self.t = 0  # Counter for time steps
        self.m = None  # First moment vector (mean)
        self.v = None  # Second moment vector (variance)

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.m is None:
            self.m = np.zeros_like(weight_tensor)  # Initialize first moment estimate
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)  # Initialize second moment estimate

        self.t += 1  # Increment time step

        # Update biased first moment estimate
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor

        # Update biased second moment estimate
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        # Correct bias for first moment
        m_hat = self.m / (1 - self.mu ** self.t)

        # Correct bias for second moment
        v_hat = self.v / (1 - self.rho ** self.t)

        # Apply Adam weight update formula
        weight_update = weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weight_update  # Return updated weights


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate  # Just the learning rate for SGD

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Simple SGD update: subtract gradient scaled by learning rate
        weight_update = weight_tensor - self.learning_rate * gradient_tensor
        return weight_update  # Return updated weights
