import datasets
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import optuna


class CTB:

    def cv_scoring(self, model, X_train, y_train, n_splits, random_state):
        scoring_list = []
        kf = KFold(n_splits=n_splits, random_state=random_state)
        for tr_cv_idx, va_cv_idx in kf.split(X_train, y_train):
            # Pool
            ptrain = Pool(
                data=X_train[tr_cv_idx],
                label=y_train[tr_cv_idx]
            )
            pvalid = Pool(
                data=X_train[va_cv_idx],
                label=y_train[va_cv_idx]
            )
            # training
            model.fit(
                ptrain,
                early_stopping_rounds=50,
                eval_set=pvalid,
                use_best_model=True
            )
            # evaluation
            y_pred = model.predict(pvalid)
            y_true = y_train[va_cv_idx]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            scoring_list.append(rmse)
        return scoring_list

    def prediction(self, model, X_train, y_train, X_valid, y_valid, X_test):
        # Pool
        ptrain = Pool(
            data=X_train,
            label=y_train
        )
        pvalid = Pool(
            data=X_valid,
            label=y_valid
        )
        ptest = Pool(
            data=X_test
        )
        # training
        model.fit(
            ptrain,
            early_stopping_rounds=50,
            eval_set=pvalid,
            use_best_model=True
        )
        # prediction
        pred_va = model.predict(pvalid)
        pred_te = model.predict(ptest)
        return pred_va, pred_te


# params initializer
params = {
    'iterations': 1000,
    'depth': 8,
    'learning_rate': 0.05,
    'random_strength': 10,
    'bagging_temperature': 1,
    'grow_policy': 'SymmetricTree',
    'eval_metric': 'RMSE',

}

# try


def obj_ctb(trial):
    space = {
        'depth': trial.suggest_int('depth', 4, 15),
        'random_strength': trial.suggest_loguniform('random_strength', 0, 3),
        'bagging_temperature': trial.suggest_loguniform(
            'bagging_temperature', 0.1, 10)
    }
    params.update(space)
    model = CatBoostRegressor(**params)
    ctb = CTB()
    score = ctb.cv_scoring(
        model,
        X_train,
        y_train,
        5,
        1607
    )
    return np.mean(score)


study = optuna.create_study()
study.optimize(obj_ctb, n_trials=2, n_jobs=-1)
