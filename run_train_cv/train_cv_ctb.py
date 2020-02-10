from load_data import load_csv
from config import setting
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

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
    ptest = Pool(
        data=X_test
    )
    # training
    model.fit(
        ptrain,
        early_stopping_rounds=20,
        eval_set=pvalid,
        use_best_model=True,
        verbose=False
    )
    # prediction
    va_pred = model.predict(
        pvalid,
        ntree_end=model.get_best_iteration()
    )
    te_pred = model.predict(
        ptest,
        ntree_end=model.get_best_iteration()
    )
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
print('CatBoost R2: ', r2)
print('CatBoost RMSE: ', rmse)

# save predictions
path = '../prediction/'
with open('{}ctb_train_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(train_preds, f)
with open('{}ctb_test_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(test_preds, f)
