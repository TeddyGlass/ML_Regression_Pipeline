import pickle
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import optuna

# train feature path
path_tr = '../prediction/*_train_preds.binaryfile'
train_path = glob.glob(path_tr)
# test feature path
path_te = '../prediction/*_test_preds.binaryfile'
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

# load y_train, y_test
y_train = pd.read_csv('../dataset/y_train.csv')
y_train = y_train.iloc[:,1]
y_train_array = np.array(y_train)

y_test = pd.read_csv('../dataset/y_test.csv')
y_test = y_test.iloc[:,1]
y_test_array = np.array(y_test)

# obj_bin
labels = np.arange(10)
y_train_bins = pd.cut(y_train, 10, labels=labels)

def obj(trial):
    solver = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-2, 20),
        'solver': trial.suggest_categorical('solver', solver),
        'max_iter': trial.suggest_int('max_iter', 10, 1000),
        'random_state': 1501
    }
    model = Ridge(**params)
    kf = StratifiedKFold(n_splits=5, random_state=1641, shuffle=True)
    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train_bins):
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
kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

va_idxes = []
va_preds = []
te_preds = []

rmse_list = []
r2_list = []

for tr_idx, va_idx in kf.split(X_train, y_train_bins):
    # train
    model.fit(X_train[tr_idx], y_train[tr_idx])
    # prediction
    va_pred = model.predict(X_train[va_idx])
    te_pred = model.predict(X_test)
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

# sort and processing prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
train_preds = np.concatenate(va_preds)[order]
test_preds = np.mean(te_preds, axis=0)

# evaluation
print('Stacking_RMSE: ', np.mean(rmse_list))
print('Stacking_R2: ', np.mean(r2_list))
print('Stacking R2 Val: ', r2_score(y_train, train_preds))
print('Stacking RMSE Val: ', np.sqrt(mean_squared_error(y_train, train_preds)))
print('Stacking R2 Test: ', r2_score(y_test, test_preds))
print('Stacking RMSE Test: ', np.sqrt(mean_squared_error(y_test, test_preds)))
print('each_RMSE')
for i in range(len(rmse_list)):
    print(rmse_list[i])
print('each_R2')
for i in range(len(r2_list)):
    print(r2_list[i])

# obs_pred plot
palette = sns.diverging_palette(220, 20, n=2)
plt.figure(figsize=(8,8))
plt.title('Stacking', fontsize=15)
plt.xlabel('y_obs', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.xlim(-4,2)
plt.ylim(-4,2)
plt.scatter(y_train, train_preds, color=palette[0])
plt.scatter(y_test, test_preds, color=palette[1])
plt.grid()
plt.show()

# sort prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
pred_valid = np.concatenate(va_preds)[order]
path = '../prediction/'
with open('{}stk_train_prediction.binaryfile'.format(path), 'wb') as f:
    pickle.dump(pred_valid, f)
