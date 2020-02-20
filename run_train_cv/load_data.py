import pandas as pd
import numpy as np


def load_csv():
    X_train = pd.read_csv('../dataset/X_train.csv')
    X_train = X_train.iloc[:, 1:]
    columns = X_train.columns
    X_train_array = np.array(X_train)

    y_train = pd.read_csv('../dataset/y_train.csv')
    y_train = y_train.iloc[:,1]
    y_train_array = np.array(y_train)

    X_test = pd.read_csv('../dataset/X_test.csv')
    X_test_array = np.array(X_test.iloc[:, 1:])

    y_test = pd.read_csv('../dataset/y_test.csv')
    y_test = y_test.iloc[:,1]
    y_test_array = np.array(y_test)

    return X_train_array, y_train_array, X_test_array, y_test_array, columns