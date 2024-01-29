import tensorflow as tf
from tensorflow import keras as kr


def gate_navigator_model(num_channels_start):
    # Inputs
    image_input = kr.layers.Input(shape=(168, 168, 1))
    tof_input = kr.layers.Input(shape=(21, 21, 1))

    # Camera Layers
    conv_camera_layer_1 = kr.layers.Conv2D(filters=num_channels_start, kernel_size=5,
                                           strides=2, padding="same")(image_input)
    bn_camera_1 = kr.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(conv_camera_layer_1)
    relu_camera_1 = kr.layers.Activation(tf.nn.relu6)(bn_camera_1)
    camera_pool_1 = kr.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(relu_camera_1)

    ## Camera DoubleBlock
    double1_conv1 = kr.layers.Conv2D(filters=num_channels_start, kernel_size=3, strides=2,
                                     padding="same")(camera_pool_1)
    double1_bn1 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double1_conv1)
    double1_relu1 = kr.layers.Activation(tf.nn.relu6)(double1_bn1)
    double1_conv2 = kr.layers.Conv2D(filters=num_channels_start, kernel_size=3, strides=1,
                                     padding="same")(double1_relu1)
    double1_bn2 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double1_conv2)
    double1_relu2 = kr.layers.Activation(tf.nn.relu6)(double1_bn2)

    # Tof Layers
    tof_layer_1 = kr.layers.Conv2D(filters=num_channels_start, kernel_size=3, strides=1,
                                   padding="same")(tof_input)
    bn_tof_1 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(tof_layer_1)
    relu_tof_1 = kr.layers.Activation(tf.nn.relu6)(bn_tof_1)

    # Concatenation
    concat = kr.layers.concatenate([double1_relu2, relu_tof_1], axis=-1)

    # DoubleBlocks 2 and 3
    double2_conv1 = kr.layers.Conv2D(filters=4*num_channels_start, kernel_size=3, strides=2,
                                     padding="same")(concat)
    double2_bn1 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double2_conv1)
    double2_relu1 = kr.layers.Activation(tf.nn.relu6)(double2_bn1)
    double2_conv2 = kr.layers.Conv2D(filters=4*num_channels_start, kernel_size=3, strides=1,
                                     padding="same")(double2_relu1)
    double2_bn2 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double2_conv2)
    double2_relu2 = kr.layers.Activation(tf.nn.relu6)(double2_bn2)
    double3_conv1 = kr.layers.Conv2D(filters=8*num_channels_start, kernel_size=3, strides=2,
                                     padding="same")(double2_relu2)
    double3_bn1 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double3_conv1)
    double3_relu1 = kr.layers.Activation(tf.nn.relu6)(double3_bn1)
    double3_conv2 = kr.layers.Conv2D(filters=8*num_channels_start, kernel_size=3, strides=1,
                                     padding="same")(double3_relu1)
    double3_bn2 = kr.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(double3_conv2)
    double3_relu2 = kr.layers.Activation(tf.nn.relu6)(double3_bn2)

    # Flatten
    fc_size = 8 * num_channels_start * 6 * 6
    flatten = kr.layers.Flatten()(double3_relu2)
    fully_connected = kr.layers.Dense(units=1, input_shape=(fc_size,))(flatten)

    return kr.models.Model(inputs=(image_input, tof_input), outputs=fully_connected)

