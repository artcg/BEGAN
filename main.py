import tensorflow as tf
import prettytensor as pt
from generator import began_generator as generator
from discriminator import began_discriminator as discriminator
from utils.misc import loadData, dataIterator
import tqdm
import numpy as np
from utils.misc import plot_gens
import time
from config import checkpoint_path, checkpoint_prefix


class BEGAN:
    loss_tracker = {'generator': [],
                    'discriminator': [],
                    'convergence_measure': []}

    def loss(D_real_in, D_real_out, D_gen_in, D_gen_out, k_t, gamma=0.75):
        '''
        The Bounrdary Equibilibrium GAN uses an approximation of the
        Wasserstein Loss between the disitributions of pixel-wise
        autoencoder loss based on the discriminator performance on
        real vs. generated data.

        This simplifies to reducing the L1 norm of the autoencoder loss:
        making the discriminator objective to perform well on real images
        and poorly on generated images; with the generator objective
        to create samples which the discriminator will perform well upon.

        args:
            D_real_in:  input to discriminator with real sample.
            D_real_out: output from discriminator with real sample.
            D_gen_in: input to discriminator with generated sample.
            D_gen_out: output from discriminator with generated sample.
            k_t: weighting parameter which constantly updates during training
            gamma: diversity ratio, used to control model equibilibrium.
        returns:
            D_loss:  discriminator loss to minimise.
            G_loss:  generator loss to minimise.
            k_tp:    value of k_t for next train step.
            convergence_measure: measure of model convergence.
        '''
        def pixel_autoencoder_loss(out, inp):
            '''
            The autoencoder loss used is the L1 norm (note that this
            is based on the pixel-wise distribution of losses
            that the authors assert approximates the Normal distribution)

            args:
                out:  discriminator output
                inp:  discriminator input
            returns:
                L1 norm of pixel-wise loss
            '''
            eta = 1  # paper uses L1 norm
            diff = tf.abs(out - inp)
            if eta == 1:
                return tf.reduce_sum(diff)
            else:
                return tf.reduce_sum(tf.pow(diff, eta))

        mu_real = pixel_autoencoder_loss(D_real_out, D_real_in)
        mu_gen = pixel_autoencoder_loss(D_gen_out, D_gen_in)
        D_loss = mu_real - k_t * mu_gen
        G_loss = mu_gen
        lam = 0.001  # 'learning rate' for k. Berthelot et al. use 0.001
        k_tp = k_t + lam * (gamma * mu_real - mu_gen)
        convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)
        return D_loss, G_loss, k_tp, convergence_measure

    def run(x, batch_size, hidden_size):
        Z = tf.random_normal((batch_size, hidden_size), 0, 1)

        with pt.defaults_scope(learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001):

            x_tilde = generator(Z, batch_size=batch_size)
            x_tilde_d = discriminator(x_tilde, batch_size=batch_size,
                                      hidden_size=hidden_size)

            x_d = discriminator(x, reuse_scope=True, batch_size=batch_size,
                                hidden_size=hidden_size)

            return x_tilde, x_tilde_d, x_d

    scopes = ['generator', 'discriminator']


def began_train(images, start_epoch=0, add_epochs=None, batch_size=16,
                hidden_size=2048, dim=(64, 64, 3), gpu_id='/gpu:0',
                demo=False, get=False, start_learn_rate=1e-5, decay_every=50,
                save_every=1, batch_norm=True, gamma=0.75):

    num_epochs = start_epoch + add_epochs
    loss_tracker = BEGAN.loss_tracker

    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        with tf.device(gpu_id):
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.AdamOptimizer(learning_rate, epsilon=1.0)

            next_batch = tf.placeholder(tf.float32,
                                        [batch_size, np.product(dim)])

            x_tilde, x_tilde_d, x_d = BEGAN.run(next_batch, batch_size,
                                                hidden_size)

            k_t = tf.placeholder(tf.float32, shape=[])
            D_loss, G_loss, k_tp, convergence_measure = \
                BEGAN.loss(next_batch, x_d, x_tilde, x_tilde_d, k_t=k_t)

            params = tf.trainable_variables()
            tr_vars = {}
            for s in BEGAN.scopes:
                tr_vars[s] = [i for i in params if s in i.name]

            G_grad = opt.compute_gradients(G_loss,
                                           var_list=tr_vars['generator'])

            D_grad = opt.compute_gradients(D_loss,
                                           var_list=tr_vars['discriminator'])

            G_train = opt.apply_gradients(G_grad, global_step=global_step)
            D_train = opt.apply_gradients(D_grad, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session(graph=graph,
                          config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=True))
        sess.run(init)
    if start_epoch > 0:
        path = '{}/{}_{}.tfmod'.format(checkpoint_path,
                                       checkpoint_prefix,
                                       str(start_epoch-1).zfill(4))
        tf.train.Saver.restore(saver, sess, path)

    k_t_ = 0  # We initialise with k_t = 0 as in the paper.
    num_batches_per_epoch = int(len(images) / batch_size)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {} / {}'.format(epoch + 1, num_epochs + 1))
        for i in tqdm.tqdm(range(num_batches_per_epoch)):
            iter_ = dataIterator([images], batch_size)

            learning_rate_ = start_learn_rate * pow(0.5, epoch // decay_every)
            next_batch_ = next(iter_)

            _, _, D_loss_, G_loss_, k_t_, M_ = \
                sess.run([G_train, D_train, D_loss, G_loss, k_tp, convergence_measure],
                         {learning_rate: learning_rate_,
                          next_batch: next_batch_, k_t: min(max(k_t_, 0), 1)})

            loss_tracker['generator'].append(G_loss_)
            loss_tracker['discriminator'].append(D_loss_)
            loss_tracker['convergence_measure'].append(M_)

        if epoch % save_every == 0:
            path = '{}/{}_{}.tfmod'.format(checkpoint_path,
                                           checkpoint_prefix,
                                           str(epoch).zfill(4))
            saver.save(sess, path)
    if demo:
        batch = dataIterator([images], batch_size).__next__()
        ims = sess.run(x_tilde)
        plot_gens((ims, batch),
                  ('Generated 64x64 samples.', 'Random training images.'),
                  loss_tracker)
        if get:
            return ims


def _train(start_epoch, train, add_epochs, max_images=50000, **k):
    SE = start_epoch
    while start_epoch <= SE + add_epochs:
        i = 0
        while True:
            images = loadData(size=max_images, offset=i)
            if train is False:
                return began_train(images, start_epoch=start_epoch,
                                   add_epochs=0, demo=True, get=True, **k)
            began_train(images, start_epoch=start_epoch, add_epochs=1,
                        **k)
            start_epoch += 1
            i += 1
            if len(images) < max_images:
                break
            del images
            time.sleep(30)  # Let my GPU cool down
        print('full cycle finished. Good time to stop.')
        time.sleep(60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run BEGAN.')

    parser.add_argument('--gpuid', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')

    parser.add_argument('--save-every', type=int, default=5,
                        help='Frequency to save checkpoint (in epochs)')

    parser.add_argument('--start-epoch', type=int, default=0, required=True,
                        help='Start epoch (0 to begin training from scratch,'
                        + 'N to restore from checkpoint N)')

    parser.add_argument('--add-epochs', type=int, default=100, required=True,
                        help='Number of epochs to train'
                        + '(-1 to train indefinitely)')

    parser.add_argument('--max-images', type=int, default=50000,
                        help='Number of images to load into RAM at once')

    parser.add_argument('--gamma', type=float, default=0.75,
                        help='Diversity ratio (read paper for more info)')

    parser.add_argument('--start-learn-rate', type=float, default=1e-5,
                        help='Starting learn rate')

    parser.add_argument('--train', type=int, default=1,
                        help='"1" to train; "0" to run and'
                        + 'return output')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default 16'
                        + 'as in paper)')

    parser.add_argument('--hidden_size', type=int, default=2048,
                        help='Dimensionality of the discriminator encoding.'
                        + '(Paper doesnt specify value so we use guess)')

    parser.add_argument('--batch-norm', type=int, default=1,
                        help='Set to "0" to disable batch normalisation')

    parser.add_argument('--decay-every', type=int, default=-1,
                        help='Number of epochs before learning rate decay'
                        + '(set to 0 to disable)')

    parser.add_argument('--outdir', type=str, default='output',
                        help='Path to save output generations')

    args = parser.parse_args()
    if args.gpuid == -1:
        args.gpuid = '/cpu:0'
    else:
        args.gpuid = '/gpu:{}'.format(args.gpuid)

    if args.decay_every == -1:
        args.decay_every = np.inf

    if not args.train:
        args.train = False

    im = _train(start_epoch=args.start_epoch, add_epochs=args.add_epochs,
                batch_size=args.batch_size, hidden_size=args.hidden_size,
                gpu_id=args.gpuid, train=args.train,
                save_every=args.save_every, decay_every=args.decay_every,
                batch_norm=args.batch_norm)

    if not args.train:
        import matplotlib.pyplot as plt
        for n in range(8):
            im_to_save = im[n].reshape([64, 64, 3])
            plt.imsave(args.outdir+'/out_{}.jpg'.format(n),
                       im_to_save)
