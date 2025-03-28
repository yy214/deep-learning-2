import numpy as np
from model import Model

def sigmoid(x):
    return 1/(1+np.exp(-x))

class RBM(Model):
    def __init__(self, p, q):
        self.a = np.random.normal(0, 0.01, p)
        self.b = np.random.normal(0, 0.01, q)
        self.W = np.random.normal(0, 0.01, (p, q))
    
    def entree_sortie(self, X):
        assert X.shape[1] == self.W.shape[0]
        return sigmoid(-X@self.W - self.b)
    
    def sortie_entree(self, H):
        assert H.shape[1] == self.W.shape[1]
        return sigmoid(H@self.W.T + self.a)
    
    def train(self, X, nb_epoch, batch_size, lr):
        n = X.shape[0]
        p, q = self.W.shape
        for epoch in range(nb_epoch):
            np.random.shuffle(X)
            for j in range(0, n, batch_size):
                X_batch = X[j:min(j+batch_size, n), :]
                real_batch_size = X_batch.shape[0]
                v0 = X_batch # batch * p
                phv0 = self.entree_sortie(v0) # batch * q
                h0 = (np.random.rand(real_batch_size, q) < phv0).astype(int)
                pvh0 = self.sortie_entree(h0) # batch * p
                v1 = (np.random.rand(real_batch_size, p) < pvh0).astype(int)
                phv1 = self.entree_sortie(v1) # batch * q
                grad_a = np.sum(v0-v1, axis=0)
                grad_b = np.sum(phv0-phv1, axis=0)
                grad_W = v0.T@phv0 - v1.T@phv1
                self.a += lr/real_batch_size * grad_a
                self.b += lr/real_batch_size * grad_b
                self.W += lr/real_batch_size * grad_W
    
    def generer_image(self, nb_data, nb_iter):
        p = self.a.shape[0]
        for i in range(nb_data):
            X_new = (np.random.rand(1, p) < 0.5).astype(int)
            for j in range(nb_iter):
                H = self.entree_sortie(X_new)
                X_new = (np.random.rand(1, p) < self.sortie_entree(H)).astype(int)
            yield X_new