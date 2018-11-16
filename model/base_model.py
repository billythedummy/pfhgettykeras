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
        encoder_stack = []
        a = segmentation_input
        # Encoder chain
        for i in range(1, self._layers):
            num_filters = int(self._startFilters * math.pow(2, i))
            squeezeFactor = int(self._squeezeFactor * math.pow(1.6817, i))

            # f(x)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)

            encoder_stack.append(a)

            # Superpixel convolution to encode image losslessly
            a = SuperpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)

        # Middle block (Part of encoder and decoder)
        num_filters = int(self._startFilters * math.pow(2, self._layers))
        squeezeFactor = int(self._squeezeFactor * math.pow(1.6817, self._layers))
        # a = ELU()(a)
        a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)
        a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)

        # Decoder chain
        for i in range(self._layers - 1, 0, -1):
            num_filters = int(self._startFilters * math.pow(2, i))
            squeezeFactor = int(self._squeezeFactor * math.pow(1.6817, i))

            # Subpixel convolution to decode image losslessly
            a = SubpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)
            # x
            a = Concatenate()([a, encoder_stack.pop()])
            # x + f(x)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same",activation="relu")(a)


        a = Conv2D(filters = self._numClasses, kernel_size=3, padding="same")(a)
        outputnode = Activation(activation=K.softmax, name="outputnode")(a)
        return Model(inputs=segmentation_input, outputs=outputnode)
