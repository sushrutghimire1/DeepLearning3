class BaseLayer:
    def __init__(self):
        self.trainable = True
        self.weights = None
        self.bias = None

    def forward(self, input_tensor):
        raise NotImplementedError

    def backward(self, output_tensor):
        raise NotImplementedError

    def initialize(self, initializer, *args):
        raise NotImplementedError
