import numpy as np
class Constant:
    def __init__(self,constant=0.1):
        self.constant= constant
        
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.full((fan_in,fan_out),self.constant)
        
class UniformRandom:
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.random.uniform(0,1,size=(fan_in,fan_out))

class Xavier:
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.random.normal(0,np.sqrt((2/(fan_in+fan_out))),size=(weights_shape))
        
        
class He:
    def initialize(self,weights_shape,fan_in, fan_out):
        return np.random.normal(0,np.sqrt((2/(fan_in))),size=(weights_shape))
