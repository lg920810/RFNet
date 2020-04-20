import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


def _hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def _return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.
    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = KL.Activation(_hard_swish)(x)
    if nl == 'RE':
        x = KL.ReLU(6.)(x)

    return x


def _squeeze(inputs):
    """Squeeze and Excitation.
    This function defines a squeeze structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
    """
    input_channels = int(inputs.shape[-1])

    x = KL.GlobalAveragePooling2D()(inputs)
    x = KL.Dense(input_channels, activation='relu')(x)
    x = KL.Dense(input_channels, activation='hard_sigmoid')(x)
    x = KL.Reshape((1, 1, input_channels))(x)
    x = KL.Multiply()([inputs, x])

    return x


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, kernel=(1, 1), strides=(1, 1), alpha=1.)

    x = KL.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)
    x = KL.ReLU(6.)(x)
    x = KL.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)

    if r:
        x = KL.Add()([x, inputs])

    return x


def _bottleneck_v3(inputs, filters, kernel, e, s, alpha=1.0, squeeze=False, nl='RE'):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[3] == filters

    x = _conv_block(inputs=inputs, filters=tchannel, kernel=(1, 1), strides=(1, 1), nl=nl)

    x = KL.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)
    x = _return_activation(x, nl)

    if squeeze:
        x = _squeeze(x)

    x = KL.Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)

    if r:
        x = KL.Add()([x, inputs])

    return x


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), alpha=1.0, nl='RE'):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides)(inputs)
    x = KL.BatchNormalization(axis=channel_axis)(x)
    return _return_activation(x, nl=nl)


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = KL.ZeroPadding2D(((0, 1), (0, 1)),
                             name='conv_pad_%d' % block_id)(inputs)
    x = KL.DepthwiseConv2D((3, 3),
                           padding='same' if strides == (1, 1) else 'valid',
                           depth_multiplier=depth_multiplier,
                           strides=strides,
                           use_bias=False,
                           name='conv_dw_%d' % block_id)(x)
    x = KL.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = KL.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = KL.Conv2D(pointwise_conv_filters, (1, 1),
                  padding='same',
                  use_bias=False,
                  strides=(1, 1),
                  name='conv_pw_%d' % block_id)(x)
    x = KL.BatchNormalization(axis=channel_axis,
                              name='conv_pw_%d_bn' % block_id)(x)
    return x


def mobilenet_graph(input_image, architecture='mobilenetv3', alpha=1., depth_multiplier=1):
    print('You are using ', architecture)
    assert architecture in ['mobilenetv1', 'mobilenetv2', 'mobilenetv3']

    if architecture == 'mobilenetv2':
        x = _conv_block(input_image, alpha=alpha, filters=32, kernel=(3, 3), strides=(2, 2))
        C1 = x
        x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)

        x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
        C2 = x
        x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
        C3 = x
        x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
        C4 = x
        x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
        x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
        x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
        C5 = x
    elif architecture == 'mobilenetv1':
        x = _conv_block(input_image, filters=32, alpha=alpha, strides=(2, 2))
        C1 = x
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=2)
        C2 = x
        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=4)
        C3 = x
        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=6)
        C4 = x
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=12)
        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
        C5 = x
    elif architecture == 'mobilenetv3_large':
        x = _conv_block(input_image, 16, (3, 3), strides=(2, 2), nl='HS')
        x = _bottleneck_v3(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = _bottleneck_v3(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = _bottleneck_v3(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = _bottleneck_v3(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = _bottleneck_v3(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = _bottleneck_v3(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = _bottleneck_v3(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = _conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')

        x = _conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = KL.GlobalAveragePooling2D()(x)
        x = KL.Reshape((1, 1, 960))(x)

        x = KL.Conv2D(1280, (1, 1), padding='same')(x)
        x = _return_activation(x, 'HS')
    else:
        x = _conv_block(input_image, 16, (3, 3), strides=(2, 2), nl='HS')
        x = _bottleneck_v3(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        C1 = x
        x = _bottleneck_v3(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = _bottleneck_v3(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        C2 = x
        x = _bottleneck_v3(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = _bottleneck_v3(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = _bottleneck_v3(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        C3 = x
        x = _bottleneck_v3(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = _bottleneck_v3(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        C4 = x
        x = _bottleneck_v3(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = _bottleneck_v3(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = _conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        C5 = x
    return [C1, C2, C3, C4, C5]


def mobilenetv1(config, input_image):
    return mobilenet_graph(input_image=input_image,
                           architecture='mobilenetv1',
                           alpha=1.,
                           depth_multiplier=1)


def mobilenetv2(config, input_image):
    return mobilenet_graph(input_image=input_image,
                           architecture='mobilenetv2',
                           alpha=1.,
                           depth_multiplier=1)


def mobilenetv3(config, input_image):
    return mobilenet_graph(input_image=input_image,
                           architecture='mobilenetv3',
                           alpha=1.,
                           depth_multiplier=1)


if __name__ == '__main__':
    input_image = KL.Input(shape=[224, 224, 3])
    model = mobilenetv3(input_image=input_image)
    model.summary()
    print()
