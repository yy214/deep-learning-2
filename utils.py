import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip

def lire_alpha_digit(char_type):
    data = scipy.io.loadmat('data/binaryalphadigs.mat')
    images_raw = data['dat']
    char_images = images_raw[char_type]
    X = np.zeros((len(char_images), len(char_images[0])*len(char_images[0][0])))
    for i, img_raw in enumerate(char_images):
        for j, line in enumerate(img_raw):
            X[i, j*len(char_images[0][0]):(j+1)*len(char_images[0][0])] = line
    return X


def lire_mnist(path = './data/MNIST/raw'):
    """Load MNIST data from `path`"""

    data = {}
    for kind in ['train', 't10k']:
        labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        
        data[kind] = (images, labels)
    
    return data['train'][0], data['train'][1], data['t10k'][0], data['t10k'][1]


def afficher_image(img, size=(20, 16)):
    r"""
    :img: 1D array
    :size: tuple
    """
    plt.imshow(img.reshape(*size), cmap='binary')
    plt.show()