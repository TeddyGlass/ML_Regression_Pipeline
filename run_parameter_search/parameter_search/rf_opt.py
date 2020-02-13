import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import optuna
from load_data import load_csv
from config import params

print("start")
# params
params = params()
params_rf = params['Regressor']['randomforest']

# data sets
X_train, y_train = load_csv()


def obj(trial):
    # define space
    space = {
        'max_features': trial.suggest_uniform(
            'max_features', 0.65, 0.95),  # 分割feature探索に用いるfeatureの数
        'max_samples': trial.suggest_uniform(
            'max_samples', 0.65, 0.95),  # ブートストラップ時のサンプリング数
        'min_impurity_decrease': trial.suggest_loguniform(
            'min_impurity_decrease', 1e-4, 1e-2),
        'min_samples_split': trial.suggest_loguniform(
            'min_samples_split', 1e-3, 1e-1)
    }
    params_rf.update(space)
    # model
    model = RandomForestRegressor(**params_rf)
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    kf = KFold(n_splits=n_splits, random_state=random_state)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        model.fit(X_train[tr_idx], y_train[tr_idx])
        # scoring
        y_pred = model.predict(X_train[va_idx])
        y_true = y_train[va_idx]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna
print('Optimize_start')
n_trials = params['Regressor']['optuna_trials']
study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=123)
)
study.optimize(obj, n_trials=n_trials, n_jobs=-1)

# parameter update
params_rf.update(study.best_params)

print('RandomForest_RMSE: ', study.best_value)
print('RandomForest_params: ', params_rf)

# save
with open('../parameter/params_rf.binaryfile', 'wb') as f:
    pickle.dump(params_rf, f)
