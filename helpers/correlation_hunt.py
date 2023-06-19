import numpy as np
import pandas as pd


def correlation_hunt(df: pd.DataFrame, eps=0.7):
    correlated_features = []
    corr = df.corr()

    for col in df.columns.tolist():
        corr_list_col = np.abs(corr[col]) > eps
        if corr_list_col.sum() > 2:
            correlated_features.append(col)

    return correlated_features
