import pickle
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

#機能追加予定
#ファイル名からモデルの名前を自動的に取得できるようにする

# train set predicitons
path_tr = '../prediction/*_train_preds.binaryfile'
train_path = glob.glob(path_tr)
# train set predicitons
path_te = '../prediction/*_test_preds.binaryfile'
test_path = glob.glob(path_te)

# train prediction feature
train_preds_list = []
for name in train_path:
    with open(name, 'rb') as f:
        train_preds = pickle.load(f)
    train_preds_list.append(train_preds)

# test prediction feature
test_preds_list = []
for name in test_path:
    with open(name, 'rb') as f:
        test_preds = pickle.load(f)
    test_preds_list.append(test_preds)
