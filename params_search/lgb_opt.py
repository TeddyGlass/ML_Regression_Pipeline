import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import optuna

# load data
from data.load_data import load_csv
from data.load_data import get_cv_index
X_train, y_train, X_test = load_csv()
tr_idx_list, va_idx_list = get_cv_index(
    X_train,
    y_train,
    5,
    2210
)


def obj(trial):
    # params
    params = {
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.65, 0.95),
        'max_depth': 7,
        'min_child_weight' = trial.suggest_loguniform('min_child_weight', 1e-4, 1e-2)
        'min_split_gain' = trial.suggest_loguniform('min_split_gain', 1e-5, 1e-1)
        'num_leaves': int(0.8*2**7)
        'subsample': trial.suggest_uniform('subsample', 0.65, 0.95),
        'learning_rate': 0.05,
        'n_estimators': 10000
    }
    model = LGBMRegressor(**params)
    kf = KFold(n_splits=5, random_state=0)
    for tr_idx, va_idx in kf.split(X_tr, y_tr):
        model.fit(
            X_tr,
            y_tr,

        )
