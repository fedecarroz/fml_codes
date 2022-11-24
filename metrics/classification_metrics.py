import numpy as np


class ClassificationMetrics:
    def __init__(self, y, pred):
        self.y = y
        self.pred = pred

        self.cm = self.confusion_matrix()
        self.tn = self.cm[0, 0]
        self.fp = self.cm[0, 1]
        self.fn = self.cm[1, 0]
        self.tp = self.cm[1, 1]

    def compute_errors(self):
        return {
            "confusion_matrix": self.cm,
            "accuracy": self.accuracy(),
            "error_rate": self.error_rate(),
            "precision": self.precision(),
            "recall": self.tp_rate(),
            "specificity": self.tn_rate(),
            "fp_rate": self.fp_rate(),
            "fn_rate": self.fn_rate(),
            "f1_score": self.f1_score(),
        }

    def confusion_matrix(self):
        m = len(self.y)
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(m):
            if self.y[i] == self.pred[i]:
                if self.pred[i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if self.pred[i] == 0:
                    fn += 1
                else:
                    fp += 1

        return np.array([[tn, fp], [fn, tp]])

    def accuracy(self):
        return (self.tp + self.tn) / self.cm.sum()

    def error_rate(self):
        return 1 - self.accuracy()

    def precision(self):
        return self.tp / (self.tp + self.fp)

    # Recall or sensitivity
    def tp_rate(self):
        return self.tp / (self.tp + self.fn)

    # Specificity
    def tn_rate(self):
        return self.tn / (self.fp + self.tn)

    def fp_rate(self):
        return 1 - self.tn_rate()

    def fn_rate(self):
        return 1 - self.tp_rate()

    def f1_score(self):
        return 2 * self.precision() * self.tp_rate() / (self.precision() + self.tp_rate())
