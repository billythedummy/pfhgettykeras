from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras.constraints import MinMaxNorm
class VariableScaling(Layer):

    def __init__(self, initializer="uniform", **kwargs):
        self.initializer = initializer
        super(VariableScaling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.k = self.add_weight(name='k',
                                      shape=[1],
                                      initializer=self.initializer,
                                      trainable=True,
                                      constraint=MinMaxNorm(min_value=-2.0, 
                                        max_value=2.0, rate=0.8))
        # Be sure to call this at the end
        super(VariableScaling, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        prev, cur = x
        
        return  [prev * self.k, cur * (1-self.k)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape

    def get_config(self):
        return {'initializer': self.initializer}