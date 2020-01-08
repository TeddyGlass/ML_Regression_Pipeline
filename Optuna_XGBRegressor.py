import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
import optuna

# data sets
boston = load_boston()
X = boston['data']
y = boston['target']

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.2)

# Objective function
def obj_xgb(trial):
    """"
    input X_train, y_train: numpy ndarray
    """
    # parameter space
    max_depth = trial.suggest_int('max_depth', 5,10)
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

    # model setting
    model = XGBRegressor(
        base_score=base_score,
        booster='gbtree',
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        colsample_bytree=colsample_bytree, 
        gamma=gamma,
        importance_type='gain', 
        learning_rate=0.05,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        n_estimators=1000,
        n_jobs=-1,
        random_state=1624,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        seed=0,
        subsample=subsample
    )
    # cross validation
    scoring_list = []
    for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
        model.fit(
            X_train[train_index],
            y_train[train_index],
            early_stopping_rounds=50,
            eval_set=(X_train[valid_index], y_train[valid_index]),
            eval_metric='rmse'
        )
        """
        * early stopping
            evaluation metrics: RMSE
        """
        # final performance evaluation
        y_true = y_train[valid_index]
        y_pred = model.predict(X_train[valid_index])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scoring_list.append(rmse)
        """
        * evaluation for optuna
            mertics: RMSE when early stopping
        """
    return np.mean(scoring_list)