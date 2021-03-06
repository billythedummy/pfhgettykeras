from keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPooling2D, Lambda
from keras.layers import Dropout, Reshape, Permute, Concatenate
from keras.layers.advanced_activations import LeakyReLU, ELU
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
    def __init__(self, start_filters=16, squeeze_factor=16, layers=4, num_classes=2, shape=(1024, 2048, 3)):
        self._startFilters = int(start_filters)
        self._squeezeFactor = int(squeeze_factor)
        self._bottleneckFilters = int(self._startFilters / self._squeezeFactor)
        self._layers = int(layers)
        self._numClasses = int(num_classes)
        self._shape = shape

    def create_model(self):
        segmentation_input = Input(
            shape=(self._shape[0], self._shape[1], 3+self._numClasses), name="segmentation_input")

        # Define encoder skip connections in a stack
        # encoder_stack = []
        a = Conv2D(filters=(self._startFilters * 2) // (4**2),
                   kernel_size=16, padding="same")(segmentation_input)
        a = ELU()(a)
        a = SuperpixelConv2D(input_shape=(
            None, None, self._startFilters), scale=4)(a)

        # encoder_stack.append(a)

        # Encoder chain
        for i in range(1, self._layers):
            num_filters = int(self._startFilters * math.pow(2, i))
            squeezeFactor = int(self._squeezeFactor * math.pow(1.6817, i))

            a = self.linear_multistep_chain(
                a, filters=num_filters, squeezeFactor=squeezeFactor,
                n=int(math.pow(2, i+1)))

            if i != self._layers - 1:
                # Linearize output into correct number of channels
                a = Conv2D(filters=(num_filters * 2)//(4**2), kernel_size=1)(a)
            else:
                 # Linearize output into correct number of channels
                a = Conv2D(filters=(num_filters)//(4**2), kernel_size=1)(a)
            # Lossless Downscale
            a = SuperpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)
            # encoder_stack.append(a)
            print(a.shape)

        # Decoder chain
        for i in range(self._layers - 1, 0, -1):
            num_filters = int(self._startFilters * math.pow(2, i))
            squeezeFactor = int(self._squeezeFactor * math.pow(1.6817, i))

            # f(x)
            a = self.linear_multistep_chain(
                a, filters=num_filters, squeezeFactor=squeezeFactor,
                n=int(math.pow(2, i)))
            # f(x) + x
            # a = Add()([a, encoder_stack.pop()])
            # Subpixel convolution to upscale/decode image
            a = Conv2D(filters=int(num_filters * (4**2) / 2), kernel_size=1,
                       )(a)
            a = SubpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)

        # Last decoder block
        # f(x)
        a = self.linear_multistep_chain(
            a, filters=self._startFilters, squeezeFactor=self._squeezeFactor,
            n=4)
        # f(x) + x
        # a = Concatenate()([a, encoder_stack.pop()])

        # Subpixel convolution to upscale/decode image
        a = Conv2D(filters=int(self._startFilters * (4**2)), kernel_size=1,
                   )(a)
        a = SubpixelConv2D(input_shape=(
            None, None, num_filters), scale=4)(a)
        # a = Reshape((self._shape[0] * self._shape[1], self._numClasses))(a)
        a = Conv2D(filters=3, kernel_size=1)(a)
        outputnode = Activation(activation=K.softmax, name="outputnode")(a)
        # reshaped = Reshape((self._shape[0], self._shape[1], self._numClasses), name="outputnode_reshaped")(outputnode)
        return Model(inputs=segmentation_input, outputs=outputnode)

    def linear_multistep_chain(self, u_0, filters, squeezeFactor, n):
        u_1 = self.residual_block(u_0, filters, squeezeFactor)
        u_nminus1 = u_0
        u_n = u_1
        for i in range(n - 1):
            # Do linear multistep
            u_nplus1 = self.linear_multistep(
                u_n, filters, squeezeFactor, u_nminus1, i + 1, n)
            # Update state
            u_nminus1 = u_n
            u_n = u_nplus1
        return u_n

    def residual_block(self, x, filters, squeezeFactor):
        # f(x)
        a = self.res_weights(x, filters, squeezeFactor)
        a = self.res_weights(a, filters, squeezeFactor)
        # f(x) + x
        a = Add()([a, x])
        return a

    def linear_multistep(self, x, filters, squeezeFactor, prev_step, block, total_blocks):
        # f(x)
        a = self.res_weights(x, filters, squeezeFactor)
        a = self.res_weights(a, filters, squeezeFactor)
        p_survival = self.get_p_survival(block=block,
                                         nb_total_blocks=total_blocks, p_survival_end=0.5, mode='linear_decay')
        a = Lambda(self.stochastic_survival, arguments={
                   'p_survival': p_survival})(a)
        # (1 - k_n) * u_n + k_n * u_n-1 = residuals
        b, c = VariableScaling()([prev_step, x])
        # f(x) + residuals
        a = Add()([a, b, c])
        return a

    def res_weights(self, x, filters, squeezeFactor):
        a = ELU()(x)
        # 1 x 1 with squeeze factor
        squeezed = Conv2D(filters=filters//squeezeFactor, kernel_size=1
                          )(a)

        # 3 x 3 Convolution
        a = Conv2D(filters=filters // 2,
                   kernel_size=3, padding="same", dilation_rate=2)(squeezed)

        # 1 x 1 Convolution
        b = Conv2D(filters=filters // 2, kernel_size=1,
                   padding="same")(squeezed)

        # # 1 x 1 with extension factor
        # a = Conv2D(filters=filters, kernel_size=1)(a)
        out = Concatenate()([a, b])
        return out

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
