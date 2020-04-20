import math
from typing import List

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file, get_source_inputs

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess

from mrcnn.efficientnet.config import BlockArgs, DEFAULT_BLOCK_LIST
from mrcnn.efficientnet.custom_objects import EfficientNetConvInitializer
from mrcnn.efficientnet.custom_objects import EfficientNetDenseInitializer
from mrcnn.efficientnet.custom_objects import Swish, DropConnect


__all__ = ['EfficientNet',
           'EfficientNetB0',
           'EfficientNetB1',
           'EfficientNetB2',
           'EfficientNetB3',
           'EfficientNetB4',
           'EfficientNetB5',
           'EfficientNetB6',
           'EfficientNetB7',
           'preprocess_input']


def preprocess_input(x, data_format=None):
    return _preprocess(x, data_format, mode='torch', backend=K)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def EfficientNet(input_shape,
                 block_args_list: List[BlockArgs],
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=None,
                 **kwargs):
    """
    Builder model for EfficientNets.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        block_args_list: Optional List of BlockArgs, each
            of which detail the arguments of the MBConvBlock.
            If left as None, it defaults to the blocks
            from the paper.
        width_coefficient: Determines the number of channels
            available per layer. Compound Coefficient that
            needs to be found using grid search on a base
            configuration model.
        depth_coefficient: Determines the number of layers
            available to the model. Compound Coefficient that
            needs to be found using grid search on a base
            configuration model.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        batch_norm_momentum: Float, default batch normalization
            momentum. Obtained from the paper.
        batch_norm_epsilon: Float, default batch normalization
            epsilon. Obtained from the paper.
        depth_divisor: Optional. Used when rounding off the coefficient
             scaled channels and depth of the layers.
        min_depth: Optional. Minimum depth value in order to
            avoid blocks with 0 layers.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.
        default_size: Specifies the default image size of the model

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = DEFAULT_BLOCK_LIST

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    x = layers.Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    feature_list = []

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        feature_list.append(x)

    # Head part
    x = layers.Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    # else:
    #     if pooling == 'avg':
    #         x = layers.GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = layers.GlobalMaxPooling2D()(x)

    outputs = x
    feature_list.append(outputs)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    # model = Model(inputs, feature_list)
    return [feature_list[0], feature_list[1], feature_list[2], feature_list[4], feature_list[7]]


def EfficientNetB0(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B0.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=224)


def EfficientNetB1(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B1.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.0,
                        depth_coefficient=1.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=240)


def EfficientNetB2(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B2.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=260)


def EfficientNetB3(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B3.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.2,
                        depth_coefficient=1.4,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=300)


def EfficientNetB4(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B4.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.4,
                        depth_coefficient=1.8,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=380)


def EfficientNetB5(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.4,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B5.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.6,
                        depth_coefficient=2.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=456)


def EfficientNetB6(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B6.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=1.8,
                        depth_coefficient=2.6,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=528)


def EfficientNetB7(config,
                   input_image,
                   input_shape=None,
                   include_top=False,
                   weights=None,
                   pooling='Max',
                   classes=1000,
                   dropout_rate=0.5,
                   drop_connect_rate=0.,
                   data_format=None):
    """
    Builds EfficientNet B7.

    # Arguments:
        input_shape: Optional shape tuple, the input shape
            depends on the configuration, with a minimum
            decided by the number of stride 2 operations.
            When None is provided, it defaults to 224.
            Considered the "Resolution" parameter from
            the paper (inherently Resolution coefficient).
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        dropout_rate: Float, percentage of random dropout.
        drop_connect_rate: Float, percentage of random droped
            connections.
        data_format: "channels_first" or "channels_last". If left
            as None, defaults to the value set in ~/.keras.

    # Raises:
        - ValueError: If weights are not in 'imagenet' or None.
        - ValueError: If weights are 'imagenet' and `classes` is
            not 1000.

    # Returns:
        A Keras Model.
    """
    return EfficientNet(input_shape,
                        DEFAULT_BLOCK_LIST,
                        width_coefficient=2.0,
                        depth_coefficient=3.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_image,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=600)


if __name__ == '__main__':

    import keras.layers as KL
    image_input =  KL.Input(shape=[224, 224, 3])
    config1 = ''
    model = EfficientNetB0(config1, image_input)

    model.summary()

