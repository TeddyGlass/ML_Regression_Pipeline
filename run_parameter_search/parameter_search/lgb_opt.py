import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import optuna
from load_data import load_csv
from config import params

# params
params = params()
params_lgb = params['Regressor']['lightgbm']

# data sets
X_train, y_train = load_csv()


def obj(trial):
    # define space
    depth = params_lgb['max_depth']
    space = {
        'num_leaves': trial.suggest_int(
            'num_leaves', 0.65*(2**depth), 0.95*(2**depth)),
        'subsample': trial.suggest_uniform('subsample', 0.65, 0.95),
        'col_sample_bytree': trial.suggest_uniform(
            'col_sample_bytree', 0.65, 0.95),
        'min_child_weight': trial.suggest_loguniform(
            'min_child_weight', 0.1, 10),
        'min_child_samples': trial.suggest_int(
            'min_child_samples', 1, X_train.shape[1]//10),
        'min_split_gain': trial.suggest_loguniform(
            'min_split_gain', 1e-4, 1e-1)
    }
    params_lgb.update(space)
    # model
    model = LGBMRegressor(**params_lgb)
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    kf = KFold(n_splits=n_splits, random_state=random_state)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        # training
        eval_set = (X_train[va_idx], y_train[va_idx])
        model.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            early_stopping_rounds=20,
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
study = optuna.create_study()
study.optimize(obj, n_trials=n_trials, n_jobs=-1)

# parameter update
params_lgb.update(study.best_params)

print('LightGBM_RMSE: ', study.best_value)
print('LightGBM_params: ', params_lgb)

# save
with open('../parameter/params_lgb.binaryfile', 'wb') as f:
    pickle.dump(params_lgb, f)
