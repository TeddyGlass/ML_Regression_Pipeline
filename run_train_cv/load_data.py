import pandas as pd
import numpy as np


def load_csv():
    path = '../dataset/'
    X_train = np.array(pd.read_csv('{}X_train.csv'.format(path)))
    y_train = np.array(pd.read_csv('{}y_train.csv'.format(path))).reshape(-1)
    X_test = np.array(pd.read_csv('{}X_test.csv'.format(path)))
    return X_train, y_train, X_test
