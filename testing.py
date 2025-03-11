from model import Model
from rbm import RBM
from principal_DBN_alpha import DBN

def test(ModelClass:Model,
         dataLoader,
         nb_epoch: int=1000,
         batch_size: int=32,
         lr: float=0.02,
         nb_data: int=1,
         nb_iter: int=10,
         *args):
    # open alphaDigit
    # specify parameters
    model = ModelClass(*args)
    X = dataLoader()
    model.train(X, nb_epoch, batch_size, lr)
    model.generer_image(nb_data, nb_iter)
