class BaseEstimator:
    def cost_func(self, m, pred, y):
        pass

    def fit(self, x_train, y_train, x_val, y_val):
        pass

    def predict(self, x):
        pass

    def compute_performance(self, x, y):
        pass

    def get_params(self):
        pass

    def set_params(self, params: dict):
        pass
