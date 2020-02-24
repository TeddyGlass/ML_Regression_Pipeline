from load_data import load_csv
from config import setting
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor

# data sets
X_train, y_train, X_test, y_test, columns = load_csv()
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)
print('len(len(columns)', len(columns))

# obj_bin
labels = np.arange(10)
y_train_bins = pd.cut(y_train, 10, labels=labels)

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
kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

va_idxes = []
va_preds = []
te_preds = []

rmse_list = []
r2_list = []

for tr_idx, va_idx in kf.split(X_train, y_train_bins):
    # training
    eval_set = (X_train[va_idx], y_train[va_idx])
    model.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        early_stopping_rounds=2,
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
print('CatBoost R2 Val: ', r2_score(y_train, train_preds))
print('CatBoost RMSE Val: ', np.sqrt(mean_squared_error(y_train, train_preds)))
print('CatBoost R2 Test: ', r2_score(y_test, test_preds))
print('CatBoost RMSE Test: ', np.sqrt(mean_squared_error(y_test, test_preds)))
print('each_RMSE')
for i in range(len(rmse_list)):
    print(rmse_list[i])
print('each_R2')
for i in range(len(r2_list)):
    print(r2_list[i])

# obs_pred plot
palette = sns.diverging_palette(220, 20, n=2)
plt.figure(figsize=(8,8))
plt.title('CatBoost', fontsize=15)
plt.xlabel('y_obs', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.xlim(-4,2)
plt.ylim(-4,2)
plt.scatter(y_train, train_preds, color=palette[0])
plt.scatter(y_test, test_preds, color=palette[1])
plt.grid()
plt.show()

# save predictions
path = '../prediction/'
with open('{}ctb_train_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(train_preds, f)
with open('{}ctb_test_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(test_preds, f)
