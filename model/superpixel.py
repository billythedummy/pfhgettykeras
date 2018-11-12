from keras.layers import Lambda


def SuperpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do space to depth convolution.

    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: downsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def superpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] == None else (input_shape[1] // scale),
                None if input_shape[2] == None else (input_shape[2] // scale),
                int(input_shape[3] * (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def superpixel(x):
        import tensorflow as tf
        return tf.space_to_depth(x, scale)

    return Lambda(superpixel, output_shape=superpixel_shape)
