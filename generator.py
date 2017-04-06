import tensorflow as tf
import prettytensor as pt
from utils import custom_ops

if False:  # This to silence pyflake
    custom_ops


def began_generator(Z, batch_size, scope_name="generator",
                    reuse_scope=False):
    '''
    The Boundary Equilibrium GAN deliberately uses a simple generator
    architecture. This is because the authors are proving that the
    excellent results are due to the innovative model design rather
    than tricks such as batch normalisation.

    This implementation uses batch normalisation by default since it
    seems to improve training

    Effectively, the generator consists of 3x3 convolutions (with ELU
    applied after each layer), combined with nearest neighbour upsampling
    to reach the desired resolution.

    Args:
        Z: Latent space
        batch_size: Batch size of generations
        scope_name: Tensorflow scope name
        reuse_scope: Tensorflow scope handling
    Returns:
        Flattened tensor of generated images, with dimensionality:
            batch_size * 64 * 64 * 3
    '''

    n = 128  # 'n' is number of filter dimensions

    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()

        layer_1 = (pt.wrap(Z)  # (hidden_size)
                   .flatten()
                   .fully_connected(8 * 8 * n, activation_fn=tf.nn.elu)
                   .fc_batch_norm()
                   .reshape([-1, 8, 8, n]))  # '-1' is batch size

        conv_1 = (layer_1
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        conv_2 = (conv_1
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        layer_2 = (conv_2
                   .apply(tf.image.resize_nearest_neighbor, [16, 16]))

        conv_3 = (layer_2
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        conv_4 = (conv_3
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        layer_3 = (conv_4
                   .apply(tf.image.resize_nearest_neighbor, [32, 32]))

        conv_5 = (layer_3
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        conv_6 = (conv_5
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        layer_4 = (conv_6
                   .apply(tf.image.resize_nearest_neighbor, [64, 64]))

        conv_7 = (layer_4
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        conv_8 = (conv_7
                  .custom_conv2d(n, k_h=3, k_w=3, d_h=1, d_w=1)
                  .conv_batch_norm()
                  .apply(tf.nn.elu))

        conv_9 = (conv_8
                  .custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1)
                  .apply(tf.sigmoid))

        return conv_9.flatten()
