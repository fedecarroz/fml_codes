import numpy as np

from estimators.base_estimator import BaseEstimator
from metrics import metrics_evaluation as m_eval


class GridSearchCV:
    def __init__(self, estimator: BaseEstimator, param_grid: dict, scoring: str, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None

    def __comb_gen(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        comb = np.array(np.meshgrid(*values), dtype=object).T
        comb = comb.reshape(round(comb.size / len(keys)), len(values))
        for r in comb:
            param = dict(sorted(zip(keys, r)))
            yield param

    def __create_folds(self, x, y):
        m = len(x)
        fold_dim = round(m / self.cv)
        for i in range(self.cv):
            first_index = i * fold_dim
            last_index = (i + 1) * fold_dim
            if last_index > m:
                last_index = m

            val_indexes = list(range(first_index, last_index))

            x_train = np.delete(x, val_indexes, axis=0)
            y_train = np.delete(y, val_indexes, axis=0)
            x_val = x[val_indexes]
            y_val = y[val_indexes]

            yield x_train, y_train.reshape(-1, ), x_val, y_val.reshape(-1, )

    def fit(self, x, y):
        for param in self.__comb_gen():
            fold_count = 1
            scores = np.array([])
            for x_train, y_train, x_val, y_val in self.__create_folds(x, y):
                self.estimator.set_params(param)
                self.estimator.fit(x_train, y_train, x_val, y_val)
                x_val_p = np.delete(x_val, 0, axis=1)
                perf = self.estimator.compute_performance(x_val_p, y_val)

                if self.scoring in perf.keys():
                    score = perf[self.scoring]
                    scores = np.append(scores, score)

                    print(f"[CV {fold_count}/{self.cv}] ... {param} ... score {score}")
                else:
                    raise Exception("No such scoring")

                fold_count += 1

            cv_score = scores.mean()

            if self.best_score_ is not None:
                if m_eval[self.scoring] == "min":
                    is_best_score = cv_score < self.best_score_
                else:
                    is_best_score = cv_score > self.best_score_
            else:
                is_best_score = False

            if self.best_score_ is None or is_best_score:
                self.best_score_ = cv_score
                self.best_params_ = param
