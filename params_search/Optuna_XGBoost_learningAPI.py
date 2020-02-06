import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import optuna

# data sets
boston = load_boston()
X = boston['data']
y = boston['target']

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.2)

# Objective function
def obj_xgb(trial):
    # parameter space
    max_depth = trial.suggest_int('max_depth', 10,100)
    gamma = trial.suggest_loguniform('gamma', 1e-5,1e2)
    min_child_weight = trial.suggest_loguniform('min_child_weight',1e-2,10)
    subsample = trial.suggest_uniform('subsample', 0.5,1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5,1.0)
    colsample_bylevel = trial.suggest_uniform('colsample_bylevel', 0.5,1.0)
    colsample_bynode = trial.suggest_uniform('colsample_bynode', 0.5,1.0)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-5,1e2)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-5,1e2)
    scale_pos_weight = trial.suggest_uniform('scale_pos_weight', 0.3,1.0)
    base_score = trial.suggest_uniform('base_score', 0.0,1.0)
    # params
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
        'min_child_weight': min_child_weight,
        'n_jobs':-1,
        'random_state':1624,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'subsample': subsample
    }  
    # train and cross validation
    scoring_list = []
    for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
        dtrain = xgb.DMatrix(X_train[train_index],label=y_train[train_index])
        dvalid = xgb.DMatrix(X_train[valid_index],label=y_train[valid_index])
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        bst = xgb.train(param,
                        dtrain,
                        num_boost_round=1000,
                        early_stopping_rounds=50,
                        evals=evals
                       )
        # final performance evaluation
        y_true = y_train[valid_index]
        y_pred = bst.predict(dvalid).round()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scoring_list.append(rmse)
    return np.mean(scoring_list)

# try optimization with optuna
study = optuna.create_study()
study.optimize(obj_xgb, n_trials=50, n_jobs=-1)

# search optimal iteration for whole data
best_params = study.best_params
param = {
    'objective': 'reg:squarederror',
    'max_depth': best_params['max_depth'],
    'booster': 'gbtree',
    'colsample_bylevel': best_params['colsample_bylevel'],
    'colsample_bynode': best_params['colsample_bynode'],
    'colsample_bytree': best_params['colsample_bytree'], 
    'gamma': best_params['gamma'],
    'importance_type': 'gain', 
    'eta':0.05,
    'min_child_weight': best_params['min_child_weight'],
    'n_jobs':-1,
    'random_state':1624,
    'reg_alpha': best_params['reg_alpha'],
    'reg_lambda': best_params['reg_lambda'],
    'subsample': best_params['subsample']
} 
iteration_list = []
for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
    dtrain = xgb.DMatrix(X_train[train_index],label=y_train[train_index])
    dvalid = xgb.DMatrix(X_train[valid_index],label=y_train[valid_index])
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=50,
        evals=evals
    )
    best_iteration = bst.best_iteration
    print('best_iteration', best_iteration)
    iteration_list.append(best_iteration)
print('mean_iteration', np.mean(iteration_list, dtype=int))

# build optimal model
bst = xgb.train(
    param,
    dtrain,
    num_boost_round=np.mean(iteration_list, dtype=int)
)

# performance evaluation
dtest = xgb.DMatrix(X_test,label=y_test)
y_true = y_test
y_pred = bst.predict(dtest).round()
r2 = r2_score(y_true, y_pred)
r2