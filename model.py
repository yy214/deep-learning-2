from abc import ABC, abstractmethod

class Model(ABC):    
    @abstractmethod
    def train(self, X, nb_epoch, batch_size, lr):
        pass

    @abstractmethod
    def generer_image(self, nb_data, nb_iter):
        pass