import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import optuna
from load_data import load_csv
from config import params

# params
params = params()
params_xgb = params['Regressor']['xgboost']

# data sets
X_train, y_train = load_csv()


def obj(trial):
    # define space
    space = {
        'subsample': trial.suggest_uniform(
            'subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_uniform(
            'colsample_bytree', 0.65, 0.95),
        'gamma': trial.suggest_loguniform(
            'gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_loguniform(
            'min_child_weight', 0.1, 10)
    }
    params_xgb.update(space)
    # model
    model = XGBRegressor(**params_xgb)
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    kf = KFold(n_splits=n_splits, random_state=random_state)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        # training
        eval_set = [(X_train[va_idx], y_train[va_idx])]
        model.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            early_stopping_rounds=20,
            eval_set=eval_set,
            eval_metric='rmse',
            verbose=0
        )
        # scoring
        y_pred = model.predict(X_train[va_idx])
        y_true = y_train[va_idx]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna
n_trials = params['Regressor']['optuna_trials']
study = optuna.create_study()
study.optimize(obj, n_trials=n_trials, n_jobs=-1)

# parameter update
params_xgb.update(study.best_params)

print('XGBoost_RMSE: ', study.best_value)
print('XGBoost_params: ', params_xgb)

# save
with open('../parameter/params_xgb.binaryfile', 'wb') as f:
    pickle.dump(params_xgb, f)
