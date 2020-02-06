import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_csv():
    X_train = np.array(pd.read_csv('./datasets/X_train.csv'))
    X_test = np.array(pd.read_csv('./datasets/X_test.csv'))
    y_train = np.array(pd.read_csv('./datasets/y_train.csv'))
    return X_train, y_train, X_test


def get_cv_index(X_train, y_train, n_splits, random_state):
    kf = KFold(n_splits=n_splits, random_state=random_state)
    tr_idx_list = [tr_idx for tr_idx, va_idx in kf.split(X_train, y_train)]
    va_idx_list = [va_idx for tr_idx, va_idx in kf.split(X_train, y_train)]
    return tr_idx_list, va_idx_list
