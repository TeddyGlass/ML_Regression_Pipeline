import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import optuna

# data sets
boston = load_boston()
X = boston['data']
y = boston['target']

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.2)

# XGboost train
def train_xgb (dtrain, dvalid,
               max_depth, gamma, min_child_weight,
               subsample, colsample_bytree, colsample_bylevel, colsample_bynode,
               reg_alpha, reg_lambda,
               scale_pos_weight,
               base_score
              ):
    param = {
        'objective': 'reg:squarederror',
        'max_depth': max_depth,
        'booster': 'gbtree',
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'colsample_bytree': colsample_bytree, 
        'gamma': gamma,
        'importance_type': 'gain', 
        'eta':0.05,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'n_jobs':-1,
        'random_state':1624,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'subsample': subsample
    }
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(param,
                    dtrain,
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    evals=evals,
                   )
    return bst

# cross validation and model training
for train_index, valid_index in KFold(n_splits=2, random_state=0).split(X_train, y_train):
    dtrain = xgb.DMatrix(X_train[train_index],label=y_train[train_index])
    dvalid = xgb.DMatrix(X_train[valid_index],label=y_train[valid_index])
    train_xgb(
        dtrain=dtrain, dvalid=dvalid,
        max_depth=3, gamma=1, min_child_weight=0.5,
        subsample=0.5, colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,
        reg_alpha=0, reg_lambda=0,
        scale_pos_weight=1,
        base_score=0.5
    )