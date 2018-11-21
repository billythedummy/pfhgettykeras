from keras.layers import Input, Dense, SeparableConv2D, Activation, Add, MaxPooling2D
from keras.layers import Dropout, Reshape, Permute, Concatenate, Lambda
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.models import Model, Sequential
import tensorflow as tf
# Custom layers
from coord import CoordinateChannel2D
from subpixel import SubpixelConv2D
from superpixel import SuperpixelConv2D
from variablescaling import VariableScaling
# Custom initialization
from icnr import ICNR
import math

import keras.backend as K


class SegmentationNetwork:
    # Layers specify amount of encoder and decoder layers (Total amount of layers will equal layers * 2!)
    def __init__(self, start_filters=16, layers=4, num_classes=2, shape=(1024, 2048, 3)):
        self._startFilters = int(start_filters)
        self._layers = int(layers)
        self._numClasses = int(num_classes)
        self._shape = shape

    def create_model(self):
        segmentation_input = Input(
            shape=(self._shape[0], self._shape[1], 3+self._numClasses), name="segmentation_input")

        # Define encoder skip connections in a stack
        encoder_stack = []
        a = segmentation_input
        # Encoder chain
        for i in range(1, self._layers):
            num_filters = int(self._startFilters * math.pow(2, i))

            # f(x)
            a = SeparableConv2D(kernel_initializer="he_uniform", filters=num_filters,
                                kernel_size=3, padding="same")(a)
            a = self.linear_multistep_chain(
                a, filters=num_filters,
                n=int(i**2.5))
            encoder_stack.append(a)

            # Superpixel convolution to encode image losslessly
            a = SuperpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)

        # Middle block (Part of encoder and decoder)
        num_filters = int(self._startFilters * math.pow(2, self._layers))
        # a = ELU()(a)
        a = SeparableConv2D(kernel_initializer="he_uniform", filters=num_filters,
                            kernel_size=3, padding="same")(a)
        a = self.linear_multistep_chain(
            a, filters=num_filters, n=self._layers ** 2)

        # Decoder chain
        for i in range(self._layers - 1, 1, -1):
            num_filters = int(self._startFilters * math.pow(2, i))

            # Subpixel convolution to decode image losslessly
            a = SubpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)
            # f(x) + x
            a = Concatenate()([a, encoder_stack.pop()])
            # f(x)
            a = SeparableConv2D(kernel_initializer="he_uniform", filters=num_filters,
                                kernel_size=1, padding="same")(a)
            a = self.linear_multistep_chain(
                a, filters=num_filters,
                n=int(i ** 1.7))

        # Subpixel convolution to decode image losslessly
        a = SubpixelConv2D(input_shape=(
            None, None, num_filters), scale=4)(a)
        # f(x) + x
        a = Concatenate()([a, encoder_stack.pop()])
        a = SeparableConv2D(kernel_initializer="he_uniform", filters=self._numClasses,
                            kernel_size=3, padding="same")(a)
        unmatted_outputnode = Activation(activation=K.softmax, name="unmatted_outputnode")(a)
        
        # Get reference to initial output
        dense = unmatted_outputnode
        rgb = self.crop(3, 0, 3)(segmentation_input)
        dense = Concatenate()([dense, rgb])
        # Densenet Matting
        for i in range(3):
            a = LeakyReLU()(dense)
            a = SeparableConv2D(kernel_initializer="he_uniform", filters=self._numClasses,
                                kernel_size=3, padding="same")(a)
            dense = Concatenate()([a, dense])

        a = SeparableConv2D(kernel_initializer="he_uniform", filters=self._numClasses, kernel_size=3,
                            padding="same")(dense)
        
        matted_outputnode = Activation(activation=K.softmax, name="matted_outputnode")(a)
        return Model(inputs=segmentation_input, 
            outputs=[unmatted_outputnode, matted_outputnode])

    def linear_multistep_chain(self, u_0, filters,  n):
        u_1 = self.residual_block(u_0, filters)
        u_nminus1 = u_0
        u_n = u_1
        for i in range(n - 1):
            # Do linear multistep
            u_nplus1 = self.linear_multistep(
                u_n, filters, u_nminus1, i, n)
            # Update state
            u_nminus1 = u_n
            u_n = u_nplus1
        return u_n

    def residual_block(self, x, filters):
        # f(x)
        a = self.res_weights(x, filters, dilation_rate=2)
        # f(x) + x
        a = Add()([a, x])
        return a

    def linear_multistep(self, x, filters, prev_step, block, total_blocks):
        # f(x)
        a = self.res_weights(x, filters, dilation_rate=2 + 4 // total_blocks)
        # p_survival = self.get_p_survival(block=block,
        #                                  nb_total_blocks=total_blocks, p_survival_end=0.7,
        #                                  mode='uniform')
        # a = Lambda(self.stochastic_survival, arguments={
        #            'p_survival': p_survival})(a)
        # (1 - k_n) * u_n + k_n * u_n-1 = residuals
        b, c = VariableScaling()([prev_step, x])
        # f(x) + residuals
        a = Add()([a, b, c])
        return a

    def res_weights(self, x, filters, dilation_rate):
        a = LeakyReLU()(x)
        # 3 x 3 Dilated Convolution
        a = SeparableConv2D(kernel_initializer="he_uniform", filters=filters,
                            kernel_size=3, padding="same",
                            dilation_rate=dilation_rate)(a)
        a = LeakyReLU()(a)
        # 3 x 3 Convolution
        a = SeparableConv2D(kernel_initializer="he_uniform", filters=filters, kernel_size=3,
                            padding="same")(a)
        # out = Concatenate()([a, b])
        return a

    def get_p_survival(self, block, nb_total_blocks, p_survival_end=0.5, mode='linear_decay'):
        """
        See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
        """
        if mode == 'uniform':
            return p_survival_end
        elif mode == 'linear_decay':
            return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)

    def stochastic_survival(self, y, p_survival=1.0):
        # binomial random variable
        survival = K.random_binomial((1,), p=p_survival)
        # during testing phase:
        # - scale y (see eq. (6))
        # - p_survival effectively becomes 1 for all layers (no layer dropout)
        return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y,
                               survival * y)

    def crop(self, dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)