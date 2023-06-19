import matplotlib.pyplot as plt

from clustering.k_means import KMeans
from clustering.k_medoids import KMedoids
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

seed = 42

proc_data, y_true = datasets.make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=seed,
)

n_clusters = 4

sc = StandardScaler()
proc_data = sc.fit_transform(proc_data)

kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(proc_data)
y_pred_1 = kmeans.predict(proc_data)
print(f"Silhouette Coefficient Home K-Means: {silhouette_score(proc_data, y_pred_1)}")

plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred_1, s=n_clusters)
plt.show()

kmedoids = KMedoids(n_clusters=n_clusters, random_state=seed)
kmedoids.fit(proc_data)
y_pred_2 = kmedoids.predict(proc_data)
print(f"Silhouette Coefficient Home K-Medoids: {silhouette_score(proc_data, y_pred_2)}")

plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred_2, s=n_clusters)
plt.show()
