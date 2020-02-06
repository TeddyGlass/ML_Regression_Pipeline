import pandas as pd
import numpy as np


def load_csv():
    X_train = np.array(pd.read_csv('../dataset/X_train.csv'))
    y_train = np.array(pd.read_csv('../dataset/y_train.csv')).reshape(-1)
    return X_train, y_train
