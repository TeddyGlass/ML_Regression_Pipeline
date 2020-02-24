import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from lightgbm import LGBMRegressor
import optuna
from load_data import load_csv
from config import params

# params
params = params()
params_lgb = params['Regressor']['lightgbm']

# data sets
X_train, y_train, columns = load_csv()
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('len(len(columns)', len(columns))

# obj_bin
labels = np.arange(10)
y_train_bins = pd.cut(y_train, 10, labels=labels)


def obj(trial):
    # define space
    num_leaves = 31
    space = {
        'num_leaves': trial.suggest_int(
            'num_leaves', num_leaves, 2*num_leaves),
        'subsample': trial.suggest_uniform('subsample', 0.65, 0.85),
        'colsample_bytree': trial.suggest_uniform(
            'colsample_bytree', 0.65, 0.95),
        'min_child_weight': trial.suggest_loguniform(
            'min_child_weight', 0.1, 10),
        'min_child_samples': trial.suggest_int(
            'min_child_samples', 1, X_train.shape[0]//20),
        'min_split_gain': trial.suggest_loguniform(
            'min_split_gain', 1e-5, 1e-2)
    }
    params_lgb.update(space)
    # model
    model = LGBMRegressor(**params_lgb)
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    # kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        # training
        eval_set = (X_train[va_idx], y_train[va_idx])
        model.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            early_stopping_rounds=15,
            eval_set=eval_set,
            eval_metric='root_mean_squared_error',
            verbose=False
        )
        # scoring
        y_pred = model.predict(
            X_train[va_idx],
            ntree_limit=model.best_iteration_)
        y_true = y_train[va_idx]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna
n_trials = params['Regressor']['optuna_trials']
study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=1541)
)
study.optimize(obj, n_trials=n_trials, n_jobs=-1)

# parameter update
params_lgb.update(study.best_params)

print('LightGBM_RMSE: ', study.best_value)
print('LightGBM_params: ', params_lgb)

# save
with open('../parameter/params_lgb.binaryfile', 'wb') as f:
    pickle.dump(params_lgb, f)
