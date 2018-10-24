from keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPooling2D, Lambda, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.applications import MobileNetV2
from coord import CoordinateChannel2D
from subpixel import SubpixelConv2D
import math


class SegmentationNetwork:
    # Layers specify amount of encoder and decoder layers (Total amount of layers will equal layers * 2!)
    def __init__(self, start_filters=16, squeeze_factor=16, layers=4, num_classes=2):
        self._startFilters = int(start_filters)
        self._squeezeFactor = int(squeeze_factor)
        self._bottleneckFilters = int(self._startFilters / self._squeezeFactor)
        self._layers = int(layers)
        self._numClasses = num_classes

    def create_model(self):
        segmentation_input = Input(
            shape=(None, None, 3+self._numClasses), name="segmentation_input")

        # Append coordinate channel to input
        a = CoordinateChannel2D(use_radius=True)(segmentation_input)

        # Beginning convolution
        a = Conv2D(filters=self._startFilters,
                   kernel_size=3, padding="same")(a)
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)

        encoder_stack = []
        # Encoder block
        for i in range(0, self._layers):
            a = self.residual_block(a, filters=int(
                self._startFilters * math.pow(2, i)))
            encoder_stack.append(a)
            a = Conv2D(filters=int(
                self._startFilters * math.pow(2, i + 1)), kernel_size=3, padding="same", strides=2)(a)
            a = BatchNormalization()(a)
            a = Activation(activation="relu")(a)

        # Decoder block
        for i in range(0, self._layers):
            num_filters = int(self._startFilters *
                              math.pow(2, self._layers - 1 - i))
            # Subpixel convolution to upscale/decode image
            a = SubpixelConv2D(input_shape=(
                None, None, num_filters), scale=2)(a)
            a = Conv2D(filters=num_filters, kernel_size=3, padding="same")(a)
            a = Activation(activation="relu")(a)
            a = Add()([a, encoder_stack.pop()])

            a = self.residual_block(a, filters=num_filters)

        a = Conv2D(filters=self._numClasses, kernel_size=3, padding="same")(a)
        outputnode = Activation(activation="sigmoid", name="outputnode")(a)
        return Model(inputs=segmentation_input, outputs=outputnode)

    def residual_block(self, x, filters):
        # 1 x 1 with squeeze factor
        a = Conv2D(filters=filters//self._squeezeFactor, kernel_size=1
                   )(x)
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)

        # 3 x 3 Convolution
        a = Conv2D(filters=filters//self._squeezeFactor,
                   kernel_size=3, padding="same")(a)

        # 1 x 1 with extension factor
        a = Conv2D(filters=filters, kernel_size=1)(a)

        # Batch normalization for better learns
        a = BatchNormalization()(a)
        a = Activation(activation="relu")(a)
        a = Conv2D(filters=filters, kernel_size=3, padding="same")(a)

        # Add residual
        a = Add()([a, x])
        return a
