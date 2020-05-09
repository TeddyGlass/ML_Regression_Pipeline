import pandas as pd
import numpy as np


file_path = '../data/bostom_data.csv'
obj_col = 'target'


def load_data():
    df = pd.read_csv(file_path)
    df_y_train = df[obj_col]
    df_x_train = df.drop(columns=[obj_col])
    x = np.array(df_x_train)
    y = np.array(df_y_train)
    return x, y


if __name__ == "__main__":
    pass
