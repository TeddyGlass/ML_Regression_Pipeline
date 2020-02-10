import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import optuna
from load_data import load_csv
from config import params

# params
params = params()
params_ctb = params['Regressor']['catboost']

# data sets
X_train, y_train = load_csv()


def obj(trial):
    # define space
    space = {
        'random_strength': trial.suggest_int(
            'random_strength', 0, 10),
        'bagging_temperature': trial.suggest_loguniform(
            'bagging_temperature', 0.1, 10)
    }
    params_ctb.update(space)
    # model
    model = CatBoostRegressor(**params_ctb)
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    kf = KFold(n_splits=n_splits, random_state=random_state)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        # Pool
        ptrain = Pool(
            data=X_train[tr_idx],
            label=y_train[tr_idx]
        )
        pvalid = Pool(
            data=X_train[va_idx],
            label=y_train[va_idx]
        )
        # training
        model.fit(
            ptrain,
            early_stopping_rounds=20,
            eval_set=pvalid,
            use_best_model=True,
            verbose=False
        )
        # evaluation
        y_pred = model.predict(
            pvalid,
            ntree_end=model.get_best_iteration()
        )
        y_true = y_train[va_idx]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna
n_trials = params['Regressor']['optuna_trials']
study = optuna.create_study()
study.optimize(obj, n_trials=n_trials, n_jobs=-1)

# parameter update
params_ctb.update(study.best_params)

print('CatBoost_RMSE: ', study.best_value)
print('CatBoost_params: ', params_ctb)

# save
with open('../parameter/params_ctb.binaryfile', 'wb') as f:
    pickle.dump(params_ctb, f)
