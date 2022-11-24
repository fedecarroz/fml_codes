class Normalization:
    def fit(self, x):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class MinMaxNormalization(Normalization):
    def __init__(self, a=0, b=1):
        self.min = None
        self.max = None
        self.a = a
        self.b = b

    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)

    def transform(self, x):
        return (((x - self.min) / (self.max - self.min)) * (self.b - self.a)) + self.a


class ZScoreNormalization(Normalization):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std
