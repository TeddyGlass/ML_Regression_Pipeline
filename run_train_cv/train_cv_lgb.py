from load_data import load_csv
from config import setting
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

# data sets
X_train, y_train, X_test = load_csv()

# params
path = '../parameter/'
with open('{}params_lgb.binaryfile'.format(path), 'rb') as f:
    params = pickle.load(f)

# model
setting = setting()
params.update(learning_rate=setting['learning_rate'])
model = LGBMRegressor(**params)

# CV
n_splits = setting['cv_folds']
random_state = setting['cv_random_state']
kf = KFold(n_splits=n_splits, random_state=random_state)

va_idxes = []
va_preds = []
te_preds = []

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
    # prediction
    va_pred = model.predict(X_train[va_idx])
    te_pred = model.predict(X_test)
    # append
    va_idxes.append(va_idx)
    va_preds.append(va_pred)
    te_preds.append(te_pred)

# sort and processing of prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
train_preds = np.concatenate(va_preds)[order]
test_preds = np.mean(te_preds, axis=0)

# R2, RMES, x-yplot
r2 = r2_score(y_train, train_preds)
rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print('LightGBM R2: ', r2)
print('LightGBM RMSE: ', rmse)

# save predictions
path = '../prediction/'
with open('{}lgb_train_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(train_preds, f)
with open('{}lgb_test_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(test_preds, f)
