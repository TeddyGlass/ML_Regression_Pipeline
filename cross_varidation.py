import numpy as np
from sklearn.model_selection import KFold

def generate_cv_index(X, y, folds, random_state):
    kf = KFold(n_splits=folds, random_state=random_state)
    train_index = [train_index for train_index, valid_index in kf.split(X, y)]
    valid_index = [valid_index for train_index, valid_index in kf.split(X, y)]
    key = ['train_index', 'valid_index']
    value = [train_index, valid_index]
    cv_dictionary = {}
    cv_dictionary.update(zip(key, value))
    return cv_dictionary