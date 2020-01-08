import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from lightgbm import LGBMRegressor
import optuna

# data sets
digits = load_digits()
X = digits['data']
y = digits['target']

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.2)

# Objective function
def obj_lgb(trial):
    """"
    input X_train, y_train: numpy ndarray
    """
    # parameter space
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5,1.0)
    max_depth = trial.suggest_int('max_depth', 5,10)
    min_child_samples = trial.suggest_int('min_child_samples', 2,100)
    min_child_weight = trial.suggest_loguniform('min_child_weight', 1e-4, 1e-2)
    min_split_gain = trial.suggest_loguniform('min_split_gain', 1e-5, 1e-3)
    num_leaves = trial.suggest_int('num_leaves', 2,2**10)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-5,1e2)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-5,1e2)
    subsample = trial.suggest_uniform('subsample', 0.5,1.0)
    # model setting
    model = LGBMRegressor(
        boosting_type='gbdt', 
        colsample_bytree=colsample_bytree,
        importance_type='split',
        learning_rate=0.05,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_child_weight=min_child_weight,
        min_split_gain=min_split_gain,
        n_estimators=1000,
        n_jobs=-1,
        num_leaves=num_leaves,
        random_state=1119,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
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
            eval_metric='root_mean_squared_error',
            verbose=False # when True, optimization failed with optuna
        )
        """
        * early stopping * 
            evaluation metrics: RMSE
        """
        # final performance evaluation
        y_true = y_train[valid_index]
        y_pred = model.predict(X_train[valid_index])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scoring_list.append(rmse)
        """
        * evaluation for optuna *
            mertics: RMSE when early stopping
        """
    return np.mean(scoring_list)

# try optimization with optuna
study = optuna.create_study()
study.optimize(obj_lgb, n_trials=50, n_jobs=-1)

# search optimal iteration for whole data
prams = study.best_params
model =  LGBMRegressor(
    boosting_type='gbdt', 
    colsample_bytree=prams['colsample_bytree'],
    importance_type='split',
    learning_rate=0.05,
    max_depth=prams['max_depth'],
    min_child_samples=prams['min_child_samples'],
    min_child_weight=prams['min_child_weight'],
    min_split_gain=prams['min_split_gain'],
    n_estimators=1000,
    n_jobs=-1,
    num_leaves=prams['num_leaves'],
    random_state=1119,
    reg_alpha=prams['reg_alpha'],
    reg_lambda=prams['reg_lambda'],
    subsample=prams['subsample']
)
iteration_list = []
for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
    model.fit(
        X_train[train_index],
        y_train[train_index],
        early_stopping_rounds=50,
        eval_set=(X_train[valid_index], y_train[valid_index]),
        eval_metric='root_mean_squared_error',
        verbose=False
        )
    best_iteration = model.best_iteration_
    print('best_iteration', best_iteration)
    iteration_list.append(best_iteration)
print('mean_iteration', np.mean(iteration_list))

# build optimal model
model_opt = model.set_params(n_estimators=np.mean(iteration_list, dtype=int))
model_opt.fit(X_train, y_train)

# performance evaluation
y_true = y_test
y_pred = model_opt.predict(X_test)
r2 = r2_score(y_true, y_pred)
r2