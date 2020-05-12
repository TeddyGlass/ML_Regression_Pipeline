import pandas as pd
import numpy as np


train_path = '../data/bostom_data.csv'
test_path = '../data/bostom_test.csv'
obj_col = 'target'


def load_data():
    df = pd.read_csv(train_path)
    df_y_train = df[obj_col]
    df_x_train = df.drop(columns=[obj_col])
    x = np.array(df_x_train)
    y = np.array(df_y_train)
    return x, y


def load_test():
    df = pd.read_csv(test_path)
    df_y_train = df[obj_col]
    df_x_train = df.drop(columns=[obj_col])
    x = np.array(df_x_train)
    y = np.array(df_y_train)
    return x, y


if __name__ == "__main__":
    pass
