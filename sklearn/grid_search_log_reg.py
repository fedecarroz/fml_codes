import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import warnings

seed = 42

bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)

df['diagnosis'] = bc.target

x = df.iloc[:, :-1].values
y = df[['diagnosis']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

warnings.filterwarnings('ignore')

log_reg = SGDClassifier(early_stopping=False, learning_rate='constant', random_state=seed)

parameters = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'eta0': [0.0001, 0.001, 0.01, 0.1]
}

clf = GridSearchCV(log_reg, param_grid=parameters, scoring='accuracy', cv=5, verbose=3)
clf.fit(x_train, y_train)

print("Tuned Hyperparameters:", clf.best_params_)
print("Score:", clf.best_score_)

log_reg.set_params(**clf.best_params_)

log_reg.fit(x_train, y_train)
log_reg.predict(x_test)
print("New score:", log_reg.score(x_test, y_test))
