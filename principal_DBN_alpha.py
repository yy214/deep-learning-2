from rbm import RBM
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from model import Model



class DBN(Model):

    def __init__(self, img_size ,sizes):
        self.sizes = sizes
        self.img_size = img_size
        self.rbm_layers = []
        for i in range(len(sizes)-1):
            self.rbm_layers.append(RBM(sizes[i], sizes[i+1]))

    #def train_DBN(self, data, nb_epoch, learning_rate, batch_size):
    def train(self, X, nb_epoch, batch_size, lr):
        input_data = X
        for rbm_layer in self.rbm_layers:
            rbm_layer.train(input_data, nb_epoch, batch_size, lr)
            input_data = rbm_layer.entree_sortie(input_data)
        return self
                
    
    def generer_image(self, num_images, num_iterations):
        generated_images = []
        for _ in range(num_images):
            sortie = (np.random.rand(1, self.sizes[-1]) < 0.5).astype(int)

            for rbm_layer in reversed(self.rbm_layers):

                for j in range(num_iterations):
                    entree = rbm_layer.sortie_entree(sortie)
                    sortie = (np.random.rand(sortie.shape[0], sortie.shape[1]) < rbm_layer.entree_sortie(entree)).astype(int)
                sortie = entree
            generated_images.append(entree)
            yield entree

        generated_images = np.array(generated_images).reshape((num_images, -1))
        
        # for i, img in enumerate(generated_images):
        #     plt.imshow(img.reshape(self.img_size))
        #     plt.title(f'Generated Image {i+1}')
        #     plt.show()
        
    
if __name__ == '__main__':

    data = scipy.io.loadmat('data/binaryalphadigs.mat')
    images = data['dat']
    flattened_images = []
    img_sizes = images[0][0].shape
    for alpha in images:
        for img in alpha:
            flattened_images.append(img.flatten() / 255.0)

    flattened_images = np.array(flattened_images)
    dbn = DBN(img_sizes, [flattened_images.shape[1], 500, 250, 100])
    dbn.train(flattened_images, nb_epoch=10, batch_size=256, lr=0.1)

    dbn.generer_image(num_iterations=1000, num_images=10)
    
