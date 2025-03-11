from rbm import RBM
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from model import Model


class DBN(Model):

    def __init__(self, sizes):
        self.sizes = sizes
        self.rbm_layers = []
        for i in range(len(sizes)-1):
            self.rbm_layers.append(RBM(sizes[i], sizes[i+1]))

    #def train_DBN(self, data, nb_epoch, learning_rate, batch_size):
    def train(self, X, nb_epoch, batch_size, lr):
        input_data = X
        for rbm_layer in self.rbm_layers:
            print(f'Training Layer {rbm_layer.W.shape[0]} -> {rbm_layer.W.shape[1]} ')
            rbm_layer.train(input_data, nb_epoch, batch_size, lr)
            input_data = rbm_layer.entree_sortie(input_data)
        return self
                
    
    def generer_image(self, num_images, num_iterations):
        generated_images = []
        for _ in range(num_images):
            sample = np.random.rand(1, self.sizes[-1])
            for rbm_layer in reversed(self.rbm_layers):
                for _ in range(num_iterations):
                    sample = rbm_layer.entree_sortie(sample)
                    sample = rbm_layer.sortie_entree(sample)
            generated_images.append(sample)
        
        generated_images = np.array(generated_images).reshape((num_images, -1))
        
        for i, img in enumerate(generated_images):
            plt.imshow(img.reshape((int(np.sqrt(self.sizes[0])), int(np.sqrt(self.sizes[0])))), cmap='gray')
            plt.title(f'Generated Image {i+1}')
            plt.show()
        
        return generated_images
    
if __name__ == '__main__':

    data = scipy.io.loadmat('data/binaryalphadigs.mat')
    images = data['dat']
    flattened_images = []
    for alpha in images:
        for img in alpha:
            flattened_images.append(img.flatten() / 255.0)

    flattened_images = np.array(flattened_images)
    dbn = DBN([flattened_images.shape[1], 500, 250, 100])
    dbn.train(flattened_images, nb_epoch=10, batch_size=256, lr=0.1)

    dbn.generer_image(num_iterations=1000, num_images=10)
    
