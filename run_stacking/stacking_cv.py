import pickle
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import optuna

# train feature path
path_tr = '../prediction/*train_preds.binaryfile'
train_path = glob.glob(path_tr)
# test feature path
path_te = '../prediction/*test_preds.binaryfile'
test_path = glob.glob(path_te)

# train prediction feature
train_preds_list = []
for name in train_path:
    with open(name, 'rb') as f:
        train_preds = pickle.load(f)
    train_preds_list.append(train_preds)

# test prediction feature
test_preds_list = []
for name in test_path:
    with open(name, 'rb') as f:
        test_preds = pickle.load(f)
    test_preds_list.append(test_preds)

# features for Stacking
X_train = np.stack(train_preds_list, axis=1)
X_test = np.stack(test_preds_list, axis=1)

# load y_train
y_train = np.array(pd.read_csv('../dataset/y_train.csv')).reshape(-1)


def obj(trial):
    solver = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-2, 20),
        'solver': trial.suggest_categorical('solver', solver),
        'max_iter': trial.suggest_int('max_iter', 10, 1000),
        'random_state': 1501
    }
    model = Ridge(**params)
    kf = KFold(n_splits=3, random_state=1506)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        model.fit(X_train[tr_idx], y_train[tr_idx])
        y_pred = model.predict(X_train[va_idx])
        y_true = y_train[va_idx]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna study
study = optuna.create_study()
study.optimize(obj, n_trials=100, n_jobs=-1)
path = '../parameter/'
with open('{}params_stk.binaryfile'.format(path), 'wb') as f:
    pickle.dump(study.best_params, f)

# CV
model = Ridge(**study.best_params, random_state=0)
kf = KFold(n_splits=5, random_state=1527)
va_preds = []
va_idxes = []
rmse_list = []
r2_list = []
for tr_idx, va_idx in kf.split(X_train, y_train):
    # train
    model.fit(X_train[tr_idx], y_train[tr_idx])
    # prediction
    va_pred = model.predict(X_train[va_idx])
    # scoring
    va_true = y_train[va_idx]
    rmse = np.sqrt(mean_squared_error(va_true, va_pred))
    r2 = r2_score(va_true, va_pred)
    # append
    va_preds.append(va_pred)
    va_idxes.append(va_idx)
    rmse_list.append(rmse)
    r2_list.append(r2)

# evaluation
print('STK_RMSE: ', np.mean(rmse_list))
print('STK_R2: ', np.mean(r2_list))

# sort prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
pred_valid = np.concatenate(va_preds)[order]
path = '../prediction/'
with open('{}stk_train_prediction.binaryfile'.format(path), 'wb') as f:
    pickle.dump(pred_valid, f)
