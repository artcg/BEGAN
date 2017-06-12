import tensorflow as tf
from utils.custom_ops import custom_fc, custom_conv2d

if False:  # This to silence pyflake
    custom_ops


def decoder(Z, batch_size, num_filters, hidden_size, image_size):
    '''
    The Boundary Equilibrium GAN deliberately uses a simple generator
    architecture.

    Upsampling is 3x3 convolutions, with nearest neighbour resizing
    to the desired resolution.

    Args:
        Z: Latent space
        batch_size: Batch size of generations
        num_filters: Number of filters in convolutional layers
        hidden_size: Dimensionality of encoding
        image_size: First dimension of generated image (must be 64 or 128)
        scope_name: Tensorflow scope name
        reuse_scope: Tensorflow scope handling
    Returns:
        Flattened tensor of generated images, with dimensionality:
            [batch_size, image_size * image_size * 3]
    '''
    layer_1 = custom_fc(Z, 8 * 8 * num_filters, scope='l1')

    layer_1 = tf.reshape(layer_1, [-1, 8, 8, num_filters])  # '-1' is batch size

    conv_1 = custom_conv2d(layer_1, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c1')
    conv_1 = tf.nn.elu(conv_1)

    conv_2 = custom_conv2d(conv_1, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c2')
    conv_2 = tf.nn.elu(conv_2)

    layer_2 = tf.image.resize_nearest_neighbor(conv_2, [16, 16])

    conv_3 = custom_conv2d(layer_2, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c3')
    conv_3 = tf.nn.elu(conv_3)

    conv_4 = custom_conv2d(conv_3, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c4')
    conv_4 = tf.nn.elu(conv_4)

    layer_3 = tf.image.resize_nearest_neighbor(conv_4, [32, 32])

    conv_5 = custom_conv2d(layer_3, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c5')
    conv_5 = tf.nn.elu(conv_5)

    conv_6 = custom_conv2d(conv_5, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c6')
    conv_6 = tf.nn.elu(conv_6)

    layer_4 = tf.image.resize_nearest_neighbor(conv_6, [64, 64])

    conv_7 = custom_conv2d(layer_4, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c7')
    conv_7 = tf.nn.elu(conv_7)

    conv_8 = custom_conv2d(conv_7, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c8')
    conv_8 = tf.nn.elu(conv_8)

    if image_size == 64:
        im = conv_8
    else:
        layer_5 = tf.image.resize_nearest_neighbor(conv_8, [128, 128])

        conv_9 = custom_conv2d(layer_5, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c9')
        conv_9 = tf.nn.elu(conv_9)

        conv_10 = custom_conv2d(conv_9, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c10')
        im = tf.nn.elu(conv_10)

    im = custom_conv2d(im, 3, k_h=3, k_w=3, d_h=1, d_w=1, scope='im')
    im = tf.sigmoid(im)
    im = tf.reshape(im, [-1, image_size * image_size * 3])
    return im

def began_generator(Z, batch_size, num_filters, hidden_size, image_size,
                    scope_name="generator", reuse_scope=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()
        return decoder(Z, batch_size, num_filters, hidden_size, image_size)

