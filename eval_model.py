from model import Model
from rbm import RBM
from dbn import DBN
from utils import afficher_image
from typing import Callable
import numpy as np

def evaluate(ModelClass:Model,
         dataLoader:Callable[[], np.ndarray],
         nb_epoch: int=100,
         batch_size: int=16,
         lr: float=0.1,
         nb_data: int=10,
         nb_iter: int=1000,
         **kwargs) -> None:
     
     model = ModelClass(**kwargs)
     X = dataLoader()
     model.train(X, nb_epoch, batch_size, lr)
     model.generer_image(nb_data, nb_iter)
     for im in model.generer_image(10, 1000):
          afficher_image(im)
