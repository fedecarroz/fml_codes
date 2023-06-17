from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        self.n_clusters = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.cluster_centers_ = []
        self.indices = []
        self.y_pred = None

    def fit(self, X):
        rint = self.rstate.randint
        self.indices = [rint(X.shape[0])]
        for _ in range(self.n_clusters - 1):
            i = rint(X.shape[0])
            while i in self.indices:
                i = rint(X.shape[0])
            self.indices.append(i)
        self.cluster_centers_ = X[self.indices, :]

        cost, self.y_pred = self.compute_cost(X, self.indices)
        new_cost = cost
        new_y_pred = self.y_pred.copy()
        new_indices = self.indices[:]
        initial = True
        while (new_cost < cost) | initial:
            initial = False
            cost = new_cost
            self.y_pred = new_y_pred
            self.indices = new_indices
            for k in range(self.n_clusters):  # per ogni cluster
                k_cluster_indices = [i for i, x in enumerate(new_y_pred == k) if x] # insieme degli indici dei campioni 
                for r in k_cluster_indices:  # per ogni campione del cluster selezionato
                    if r not in self.indices:
                        # se il campione selezionato non è il centroide del cluster faccio lo swap temporaneo
                        indices_temp = self.indices[:]
                        indices_temp[k] = r
                        new_cost_temp, y_pred_temp = self.compute_cost(X, indices_temp)
                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp
                            new_y_pred = y_pred_temp
                            new_indices = indices_temp

        self.cluster_centers_ = X[self.indices, :]

    def compute_cost(self, X, indices):
        y_pred = np.argmin(self.dist(X, X[indices, :]), axis=1)
        # X[y_pred == i] => insieme dei punti del cluster i
        # X[[indices[i]], :] => medoide del cluster i
        return np.sum(
            [
                np.sum(self.dist(X[y_pred == i], X[[indices[i]], :])) for i in set(y_pred)
            ]
        ), y_pred

    def predict(self, X):
        return np.argmin(self.dist(X, self.cluster_centers_), axis=1)
