from keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPooling2D, Lambda
from keras.layers import Dropout, BatchNormalization, Reshape, Permute
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from coord import CoordinateChannel2D
from subpixel import SubpixelConv2D
import math


class SegmentationNetwork:
    # Layers specify amount of encoder and decoder layers (Total amount of layers will equal layers * 2!)
    def __init__(self, start_filters=16, squeeze_factor=16, layers=4, num_classes=2, shape=(1024, 2048, 3)):
        self._startFilters = int(start_filters)
        self._squeezeFactor = int(squeeze_factor)
        self._bottleneckFilters = int(self._startFilters / self._squeezeFactor)
        self._layers = int(layers)
        self._numClasses = num_classes
        self._shape = shape

    def create_model(self):
        segmentation_input = Input(
            shape=(self._shape[0], self._shape[1], 3+self._numClasses), name="segmentation_input")

        # Append coordinate channel to input
        # a = CoordinateChannel2D(use_radius=True)(segmentation_input)
        encoder_stack = []

        # Beginning convolution
        a = Conv2D(filters=self._startFilters,
                   kernel_size=9, padding="same", strides=4)(segmentation_input)
        a = Activation(activation="relu")(a)
        a = BatchNormalization()(a)
        encoder_stack.append(a)

        # Encoder block
        for i in range(0, self._layers):
            a = self.residual_block(a, filters=int(
                self._startFilters * math.pow(2, i)))
            encoder_stack.append(a)
            a = Conv2D(filters=int(
                self._startFilters * math.pow(2, i + 1)), kernel_size=9, padding="same", strides=4)(a)
            a = Activation(activation="relu")(a)
            a = BatchNormalization()(a)

        # Decoder block
        for i in range(0, self._layers):
            num_filters = int(self._startFilters *
                              math.pow(2, self._layers - 1 - i))
            # Subpixel convolution to upscale/decode image
            a = SubpixelConv2D(input_shape=(
                None, None, num_filters), scale=4)(a)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same")(a)
            a = Activation(activation="relu")(a)
            a = BatchNormalization()(a)
            a = Add()([a, encoder_stack.pop()])

            a = self.residual_block(a, filters=num_filters)

        a = self.upscale_block(a, residual=encoder_stack.pop(),
                            filters=self._startFilters * i, scale=4)

        a = Conv2D(filters=self._numClasses, kernel_size=1, padding="same")(a)
        a = Reshape((self._shape[0] * self._shape[1], self._numClasses))(a)
        outputnode = Activation(activation="softmax", name="outputnode")(a)
        # outputnode_formatted = Permute((2,1))(outputnode)
        # outputnode_formatted = Reshape((256, 512, self._numClasses))(outputnode_formatted)
        return Model(inputs=segmentation_input, outputs=outputnode)

    def upscale_block(self, x, residual, filters, scale):
        a = Add()([x, residual])
        a = Conv2D(filters=filters, kernel_size=3, padding="same")(a)
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)
        a = SubpixelConv2D(input_shape=(
            None, None, self._startFilters), scale=scale)(a)
        return a

    def residual_block(self, x, filters):
        # 1 x 1 with squeeze factor
        a = Conv2D(filters=filters//self._squeezeFactor, kernel_size=1
                   )(x)
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)

        # 3 x 3 Convolution
        a = Conv2D(filters=filters//self._squeezeFactor,
                   kernel_size=3, padding="same", dilation_rate=2)(a)

        # 1 x 1 with extension factor
        a = Conv2D(filters=filters, kernel_size=1)(a)

        # Batch normalization for better learns
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)
        a = Conv2D(filters=filters, kernel_size=3, padding="same")(a)

        # Add residual
        a = Add()([a, x])
        return a
