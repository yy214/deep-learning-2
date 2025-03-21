import scipy
import numpy as np
import matplotlib.pyplot as plt

def lire_alpha_digit(char_type):
    data = scipy.io.loadmat('data/binaryalphadigs.mat')
    images_raw = data['dat']
    char_images = images_raw[char_type]
    X = np.zeros((len(char_images), len(char_images[0])*len(char_images[0][0])))
    for i, img_raw in enumerate(char_images):
        for j, line in enumerate(img_raw):
            X[i, j*len(char_images[0][0]):(j+1)*len(char_images[0][0])] = line
    return X

def afficher_image(img, size=(20, 16)):
    r"""
    :img: 1D array
    :size: tuple
    """
    plt.imshow(img.reshape(*size))
    plt.show()
