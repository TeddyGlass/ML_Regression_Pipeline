from load_data import load_csv
from config import setting
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

# data sets
X_train, y_train, X_test = load_csv()

# params
path = '../parameter/'
with open('{}params_ctb.binaryfile'.format(path), 'rb') as f:
    params = pickle.load(f)

# model
setting = setting()
params.update(learning_rate=setting['learning_rate'])
model = CatBoostRegressor(**params)

# CV
n_splits = setting['cv_folds']
random_state = setting['cv_random_state']
kf = KFold(n_splits=n_splits, random_state=random_state)

va_idxes = []
va_preds = []
te_preds = []

rmse_list = []
r2_list = []

for tr_idx, va_idx in kf.split(X_train, y_train):
    # training
    eval_set = (X_train[va_idx], y_train[va_idx])
    model.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        early_stopping_rounds=20,
        eval_set=eval_set,
        use_best_model=True,
        verbose=False
    )
    # prediction
    va_pred = model.predict(
        X_train[va_idx],
        ntree_end=model.get_best_iteration()
    )
    te_pred = model.predict(
        X_test,
        ntree_end=model.get_best_iteration()
    )
    # evaluation
    va_true = y_train[va_idx]
    rmse = np.sqrt(mean_squared_error(va_true, va_pred))
    r2 = r2_score(va_true, va_pred)
    # append
    rmse_list.append(rmse)
    r2_list.append(r2)
    va_idxes.append(va_idx)
    va_preds.append(va_pred)
    te_preds.append(te_pred)

# sort and processing of prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
train_preds = np.concatenate(va_preds)[order]
test_preds = np.mean(te_preds, axis=0)

# R2, RMES
print('CatBoost R2: ', np.mean(r2_list))
print('CatBoost RMSE: ', np.mean(rmse_list))

# save predictions
path = '../prediction/'
with open('{}ctb_train_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(train_preds, f)
with open('{}ctb_test_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(test_preds, f)
