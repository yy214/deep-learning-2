from model import Model
from rbm import RBM
from dbn import DBN
import numpy as np

def calcul_softmax(rbm:RBM, entree):
    sortie = rbm.entree_sortie(entree)
    exp_sortie = np.exp(sortie)
    return exp_sortie / np.sum(exp_sortie)

def to_one_hot(arr, class_count):
    n = len(arr)
    result = np.zeros((n, class_count))
    for i in range(n):
        result[i, arr[i]] = 1
    return result

def get_predictions(arr):
    return np.argmax(arr, axis=1)

def bce_loss(y_truth, y_pred):
    return -(y_truth*np.log(y_pred) + (1-y_truth)*np.log(1-y_pred))

def activation_derivative(x):
    return -np.exp(x) / (1+np.exp(x)) # note the + sign because of our definition of the rbm

class DNN():
    def __init__(self, sizes):
        assert len(sizes) >= 2, "Need to specify at least input and output sizes"
        self.net = DBN(sizes)

    def pretrain(self, X, nb_epoch, batch_size, lr):
        self.net.train(X, nb_epoch, batch_size, lr)

    def entree_sortie_reseau(self, X):
        sorties = [X]
        for layer in self.net.rbm_layers:
            sorties.append(layer.entree_sortie(sorties[-1]))
        return sorties, calcul_softmax(self.net.rbm_layers[-1], sorties[-2])
    
    def retropropagation(self, X, labels, nb_epoch, batch_size, lr, verbose_interval = 10):
        n = X.shape[0]
        layer_count = len(self.net.rbm_layers)  # note that the last layer is the prediction layer
        y = to_one_hot(labels, self.net.rbm_layers[-1].W.shape[1])
        for ep in range(nb_epoch):
            ep_loss = 0
            indexes = np.array(range(n))
            np.random.shuffle(indexes)
            for j in range(0, n, batch_size):
                batch_ids = indexes[j:min(j+batch_size, n)]
                l = batch_ids.shape[0]
                batch_y = y[batch_ids, :]
                batch_X = X[batch_ids, :]
                sorties, y_pred = self.entree_sortie_reseau(batch_X)
                loss = bce_loss(batch_y, y_pred)
                dA = 0 # define dA but not use it yet
                # other layers
                for iLayer in range(layer_count-1, -1, -1):
                    x = sorties[iLayer]
                    curr_layer = self.net.rbm_layers[iLayer]
                    if iLayer == layer_count - 1: # softmax
                        dZ = y_pred - batch_y
                    else: # sigmoid(-entree@W - b)
                        dZ = dA * activation_derivative(x@curr_layer.W + curr_layer.b)
                    dW = 1/l * x.T @ dZ
                    db = 1/l * np.sum(dZ, axis=0)
                    dA = dZ @ curr_layer.W.T

                    # gradients
                    curr_layer.b -= lr * db
                    curr_layer.W -= lr * dW

                ep_loss += np.sum(loss)

            if verbose_interval and ep % verbose_interval == 0:
                print("Loss at episode %s: %f" % (ep, ep_loss))

    def test(self, X, y):
        _, y_pred_softmax = self.entree_sortie_reseau(X)
        y_pred = get_predictions(y_pred_softmax)
        taux_erreur = np.sum(y_pred != y) / y_pred.shape[0]
        print("Taux d'erreur :", taux_erreur.item())