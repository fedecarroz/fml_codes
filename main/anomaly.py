import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

seed = 42
warnings.filterwarnings("ignore")

df = pd.read_csv("cardio.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train = df[df.y == 0].iloc[:, :-1].values
X_test = df[df.y == 1].iloc[:, :-1].values

tsne = TSNE(n_components=2, random_state=seed)
data_viz = tsne.fit_transform(X)

plt.scatter(data_viz[:, 0], data_viz[:, 1], c=y)
plt.show()

gmm = GaussianMixture(
    n_components=10,
    random_state=seed,
    covariance_type="full",
)
gmm.fit(X_train)

pi = gmm.weights_

training_pdf = gmm.predict_proba(X_train)
test_pdf = gmm.predict_proba(X_test)
training_likelihood = pi.dot(training_pdf.T)
test_likelihood = pi.dot(test_pdf.T)

eps = 0.3
pred = test_likelihood < eps

print(f"Score: {sum(pred) / len(test_likelihood)}")
