import principal_RBM_alpha.RBM as rbm
import numpy as np
import matplotlib.pyplot as plt

class DBN:

    def init_DBN(self, sizes):
        self.sizes = sizes
        self.rbm_layers = []
        for i in range(len(sizes)-1):
            self.rbm_layers.append(rbm.init_RBM(sizes[i], sizes[i+1]))

    def train_DBN(self, data, num_iterations, learning_rate, batch_size):
        input_data = data
        for rbm_layer in self.rbm_layers:
            rbm_layer.train_RBM(input_data, num_iterations, learning_rate, batch_size)
            input_data = rbm_layer.entree_sortie_RBM(input_data)
        return self
    
    def generer_image_DBN(self, num_iterations, num_images):
        generated_images = []
        for _ in range(num_images):
            sample = np.random.rand(1, self.sizes[0])
            for rbm_layer in reversed(self.rbm_layers):
                for _ in range(num_iterations):
                    sample = rbm_layer.entree_sortie_RBM(sample)
                    sample = rbm_layer.sortie_entree_RBM(sample)
            generated_images.append(sample)
        
        generated_images = np.array(generated_images).reshape((num_images, -1))
        
        for i, img in enumerate(generated_images):
            plt.imshow(img.reshape((int(np.sqrt(self.sizes[0])), int(np.sqrt(self.sizes[0])))), cmap='gray')
            plt.title(f'Generated Image {i+1}')
            plt.show()
        
        return generated_images