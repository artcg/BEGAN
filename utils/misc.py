import numpy as np
import scipy.misc
from glob import glob
from config import data_path


def loadData(size):
    import h5py
    with h5py.File(data_path, 'r') as hf:
        faces = hf['images']
        full_size = len(faces)
        choice = np.random.choice(full_size, size, replace=False)
        faces = faces[sorted(choice)]
        faces = np.array(faces, dtype=np.float16)
        return faces / 255


def loadJPGs(path='/home/arthur/devel/input/', width=64, height=64):
    filenames = glob(path+"*.jpg")
    filenames = np.sort(filenames)

    def imread(path):
        return scipy.misc.imread(path)

    def scaleHeight(x, height=64):
        h, w = x.shape[:2]
        return scipy.misc.imresize(x, [height, int((float(w)/h)*height)])

    def cropSides(x, width=64):
        w = x.shape[1]
        j = int(round((w - width)/2.))
        return x[:, j:j+width, :]

    def get_image(image_path, width=64, height=64):
        return cropSides(scaleHeight(imread(image_path), height=height),
                         width=width)

    images = np.zeros((len(filenames), width * height * 3), dtype=np.uint8)

    for n, i in enumerate(filenames):
        im = get_image(i)
        images[n] = im.flatten()
    images = np.array(images, dtype=np.float16)
    return images / 255


def dataIterator(data, batch_size):
    '''
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    '''
    batch_idx = 0
    while True:
        length = len(data[0])
        assert all(len(i) == length for i in data)
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)
        for batch_idx in range(0, length, batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = data[0][cur_idxs]
            # images_batch = images_batch.astype("float32")
            yield images_batch


def create_image(im):
    '''
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    '''
    d1 = int(np.sqrt((np.product(im.shape) / 3)))
    im = np.array(im, dtype=np.float32)
    return np.reshape(im, (d1, d1, 3))


def plot_gens(images, rowlabels, losses):
    '''
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    '''
    import matplotlib.pyplot as plt

    examples = 8
    fig, ax = plt.subplots(nrows=len(images), ncols=examples, figsize=(18, 8))
    for i in range(examples):
        for j in range(len(images)):
            ax[(j, i)].imshow(create_image(images[j][i]), cmap=plt.cm.gray,
                              interpolation='nearest')
            ax[(j, i)].axis('off')
    title = ''
    for i in rowlabels:
        title += ' {}, '.format(i)
    fig.suptitle('Top to Bottom: {}'.format(title))
    plt.show()
    #fig.savefig(''.join(['imgs/test_',str(epoch).zfill(4),'.png']),dpi=100)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), linewidth = 4)

    D_plt, = plt.semilogy((losses['discriminator']), linewidth=4, ls='-',
                          color='b', alpha=.5, label='D')
    G_plt, = plt.semilogy((losses['generator']), linewidth=4, ls='-',
                          color='k', alpha=.5, label='G')

    plt.gca()
    leg = plt.legend(handles=[D_plt, G_plt],
                     fontsize=20)
    leg.get_frame().set_alpha(0.5)
    plt.show()
