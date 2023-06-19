import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers.grid_search_cv import GridSearchCV
from estimators.neural_network.fbgd_reg_neural_network import NeuralNetwork
from metrics.metrics_evaluation import metrics_evaluation as m_eval

np.random.seed(42)

X, y = make_regression(
    n_samples=1000,
    n_features=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

layers_list = [
    [X_train.shape[1], 4, 4, 1],
    [X_train.shape[1], 8, 8, 1]
]

nn = NeuralNetwork(
    layers=layers_list[0],
    epochs=700,
)

scoring = 'rmse'

best_score_list = []
best_params_list = []
best_layers_list = []

for layers in layers_list:
    print(layers)
    param = {'layers': layers}
    nn.set_params(param)

    parameters = {
        'alpha': [0.001, 0.01, 0.1],
        'lmd': [0.01, 0.1, 1]
    }

    gs = GridSearchCV(
        estimator=nn,
        param_grid=parameters,
        scoring=scoring,
        cv=5,
    )
    gs.fit(X_train, y_train)

    best_layers_list.append(layers)
    best_params_list.append(gs.best_params_)
    best_score_list.append(gs.best_score_)

best_score_list = np.array(best_score_list)
best_params_list = np.array(best_params_list)
best_layers_list = np.array(best_layers_list)

if m_eval[scoring] == "min":
    best_score_ = best_score_list.min(axis=0)
    pos = best_score_list.argmin()
else:
    best_score_ = best_score_list.max(axis=0)
    pos = best_score_list.argmax()

best_params_ = best_params_list[pos]
best_layers_ = best_layers_list[pos]

print(f"Tuned hyperparameters: {best_params_}")
print(f"Best layers config: {best_layers_}")
print(f"Best score: {best_score_}")

nn.set_params(best_params_)
nn.fit(X_train, y_train)

perf = nn.compute_performance(X_test, y_test)
for key, value in perf.items():
    print(f"{key}: {value}")
print(f"Last loss value: {nn.loss[-1]}")
nn.plot_loss()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
c_h, c_h_v = nn.learning_curves(X_train, y_train, X_val, y_val)
nn.plot_learning_curves(c_h, c_h_v)
