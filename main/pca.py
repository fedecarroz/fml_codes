import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from helpers.outliers_hunt import outlier_hunt
from helpers.correlation_hunt import correlation_hunt
import warnings

seed = 42
warnings.filterwarnings("ignore")

df = pd.read_csv("../datasets/glass.csv")

print(df.shape)
print(df.head(5))
print(df.describe())

features = df.columns[:-1].tolist()

outliers = outlier_hunt(df[features])
print(f"The dataset has {len(outliers)} outliers.")
df = df.drop(outliers).reset_index(drop=True)

corr_features = correlation_hunt(df[features])
print(f"The dataset has {len(corr_features)} correlated features.")
df = df.drop(columns=corr_features).reset_index(drop=True)

X = df[features]
y = df['Type']

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(random_state=seed)
pca.fit(X_train)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

n_components = None
threshold = 0.9

for i, cum_sum in enumerate(cum_var_exp):
    print(f"PC{i + 1} Cumulative variance: {cum_sum * 100} %")
    if cum_sum >= threshold and n_components is None:
        n_components = i + 1

pca_params = {"n_components": n_components}
pca.set_params(**pca_params)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

parameters = {
    'C': [10, 100],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'tol': [1e-3, 1e-4]
}

gs = GridSearchCV(
    estimator=SVC(random_state=seed),
    param_grid=parameters,
    scoring='accuracy',
    cv=kfold,
    verbose=3,
    n_jobs=-1
)
gs.fit(X_train, y_train)

print(f"BEST SCORE:\n{gs.best_score_ * 100} %")
print(f"BEST PARAMS:\n{gs.best_params_}")

gs.best_estimator_.fit(X_train, y_train)
pred = gs.best_estimator_.predict(X_test)
print(f"\nPREDICTIONS:\n{pred}")

train_sizes, train_scores, test_scores = learning_curve(
    estimator=gs.best_estimator_,
    X=X_train,
    y=y_train,
    random_state=seed,
    cv=kfold,
    scoring="accuracy",
    verbose=1,
    train_sizes=np.arange(0.1, 1.1, 0.1),
    n_jobs=-1,
)

print(f"TRAIN SIZES:\n{train_sizes}")
print(f"TRAIN SCORES:\n{train_scores}")
print(f"TEST SCORES:\n{test_scores}")

y_pred = gs.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"\nConfusion matrix:\n{cm}\n\n{report}")
