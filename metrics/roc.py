import numpy as np

from metrics.classification_metrics import ClassificationMetrics


class Roc:
    def __init__(self, y, pred):
        self.y = y
        self.pred = pred

    def roc_curve(self, t_len=100):
        thresholds = np.flip(np.linspace(0, 1, num=t_len))
        fpr = np.zeros(t_len)
        tpr = np.zeros(t_len)
        for i in range(t_len):
            new_pred = np.copy(self.pred)
            for j in range(len(self.pred)):
                if self.pred[j] > thresholds[i]:
                    new_pred[j] = 1
                else:
                    new_pred[j] = 0

            new_metrics = ClassificationMetrics(self.y, new_pred)
            fpr[i] = new_metrics.fp_rate()
            tpr[i] = new_metrics.tp_rate()

        return fpr, tpr, thresholds

    @staticmethod
    def auc(fpr, tpr):
        return np.trapz(tpr, fpr)
