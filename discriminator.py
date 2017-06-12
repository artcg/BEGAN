import tensorflow as tf
from utils.custom_ops import custom_fc, custom_conv2d
from generator import decoder

if False:  # This to silence pyflake
    custom_ops


def began_discriminator(D_I, batch_size, num_filters, hidden_size, image_size,
                        scope_name="discriminator", reuse_scope=False):
    '''
    Unlike most generative adversarial networks, the boundary
    equilibrium uses an autoencoder as a discriminator.

    For simplicity, the decoder architecture is the same as the generator.

    Downsampling is 3x3 convolutions with a stride of 2.
    Upsampling is 3x3 convolutions, with nearest neighbour resizing
    to the desired resolution.

    Args:
        D_I: a batch of images [batch_size, 64 x 64 x 3]
        batch_size: Batch size of encodings
        num_filters: Number of filters in convolutional layers
        hidden_size: Dimensionality of encoding
        image_size: First dimension of generated image (must be 64 or 128)
        scope_name: Tensorflow scope name
        reuse_scope: Tensorflow scope handling
    Returns:
        Flattened tensor of re-created images, with dimensionality:
            [batch_size, image_size * image_size * 3]
    '''


    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()

        layer_1 = tf.reshape(D_I, [-1, image_size, image_size, 3])  # '-1' is batch size

        conv_0 = custom_conv2d(layer_1, 3, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec0')
        conv_0 = tf.nn.elu(conv_0)

        conv_1 =custom_conv2d(conv_0, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec1')
        conv_1 = tf.nn.elu(conv_1)


        conv_2 =custom_conv2d(conv_1, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec2')
        conv_2 = tf.nn.elu(conv_2)


        layer_2 = custom_conv2d(conv_2, 2 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el2')
        layer_2 = tf.nn.elu(layer_2)

        conv_3 = custom_conv2d(layer_2, 2 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec3')
        conv_3 = tf.nn.elu(conv_3)

        conv_4 =custom_conv2d(conv_3, 2 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec4')
        conv_4 = tf.nn.elu(conv_4)


        layer_3 = custom_conv2d(conv_2, 3 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el3')
        layer_3 = tf.nn.elu(layer_3)

        conv_5 = custom_conv2d(layer_3, 3 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec5')
        conv_5 = tf.nn.elu(conv_5)

        conv_6 =custom_conv2d(conv_5, 3 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec6')
        conv_6 = tf.nn.elu(conv_6)


        layer_4 = custom_conv2d(conv_6, 4 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el4')
        layer_4 = tf.nn.elu(layer_4)

        conv_7 = custom_conv2d(layer_4, 4 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec7')
        conv_7 = tf.nn.elu(conv_7)

        conv_8 =custom_conv2d(conv_7, 4 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec8')
        conv_8 = tf.nn.elu(conv_8)

        if image_size == 64:
            enc = custom_fc(conv_8, hidden_size, scope='enc')
        else:
            layer_5 = custom_conv2d(conv_8, 5 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el5')
            layer_5 = tf.nn.elu(layer_5)

            conv_9 = custom_conv2d(layer_5, 5 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec9')
            conv_9 = tf.nn.elu(conv_9)

            conv_10 =custom_conv2d(conv_9, 5 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec10')
            conv_10 = tf.nn.elu(conv_10)
            enc = custom_fc(conv_10, hidden_size, scope='enc')

        # add elu before decoding?
        return decoder(enc, batch_size=batch_size, num_filters=num_filters,
                       hidden_size=hidden_size, image_size=image_size)
