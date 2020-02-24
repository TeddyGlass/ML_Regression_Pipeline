from load_data import load_csv
from config import setting
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor

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
with open('{}params_xgb.binaryfile'.format(path), 'rb') as f:
    params = pickle.load(f)

# model
setting = setting()
params.update(learning_rate=setting['learning_rate'])
model = XGBRegressor(**params)

# CV
n_splits = setting['cv_folds']
random_state = setting['cv_random_state']
# kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
va_idxes = []
va_preds = []
te_preds = []

rmse_list = []
r2_list = []
model_list = []

for tr_idx, va_idx in kf.split(X_train, y_train):
    # training
    eval_set = [(X_train[tr_idx], y_train[tr_idx]), (X_train[va_idx], y_train[va_idx])]
    model.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        early_stopping_rounds=30,
        eval_set=eval_set,
        eval_metric='rmse',
        verbose=0
    )
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
    model_list.append(model)
    va_idxes.append(va_idx)
    va_preds.append(va_pred)
    te_preds.append(te_pred)

# sort and processing of prediction instance
valid_index = np.concatenate(va_idxes)
order = np.argsort(valid_index)
train_preds = np.concatenate(va_preds)[order]
test_preds = np.mean(te_preds, axis=0)

# R2, RMES
print('XGBoost R2 Val: ', r2_score(y_train, train_preds))
print('XGBoost RMSE Val: ', np.sqrt(mean_squared_error(y_train, train_preds)))
print('XGBoost R2 Test: ', r2_score(y_test, test_preds))
print('XGBoost RMSE Test: ', np.sqrt(mean_squared_error(y_test, test_preds)))
print('each_RMSE')
for i in range(len(rmse_list)):
    print(rmse_list[i])
print('each_R2')
for i in range(len(r2_list)):
    print(r2_list[i])

# obs_pred plot
palette = sns.diverging_palette(220, 20, n=2)
plt.figure(figsize=(8,8))
plt.title('XGboost', fontsize=15)
plt.xlabel('y_obs', fontsize=15)
plt.ylabel('y_pred', fontsize=15)
plt.xlim(-4,2)
plt.ylim(-4,2)
plt.scatter(y_train, train_preds, color=palette[0])
plt.scatter(y_test, test_preds, color=palette[1])
plt.grid()
plt.show()

# learning analysis
# learning cureve plot
def show_learning_curve(train_rmse, valid_rmse):
    palette = sns.diverging_palette(220, 20, n=2)
    width = np.arange(train_rmse.shape[0])
    plt.figure(figsize=(10,7.32))
    plt.title('Learning_Curve', fontsize=15)
    plt.xlabel('Estimators', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.plot(width, train_rmse, label='train_ramse', color=palette[0])
    plt.plot(width, valid_rmse, label='valid_rmse', color=palette[1])
    plt.legend(loc='upper right', fontsize=13)
    plt.show()
# learning corve
for model in model_list:
    train_rmse = np.array(model.evals_result_['validation_0']['rmse'])
    valid_rmse = np.array(model.evals_result_['validation_1']['rmse'])
    show_learning_curve(train_rmse, valid_rmse)

# save predictions
path = '../prediction/'
with open('{}xgb_train_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(train_preds, f)
with open('{}xgb_test_preds.binaryfile'.format(path), 'wb') as f:
    pickle.dump(test_preds, f)

# importance analysis
importance_list = [] 
for i in range(len(model_list)):
    model = model_list[i]
    importance = model.feature_importances_
    importance_list.append(importance)

# sort feature_importance
imporatance_ave = np.mean(np.stack(importance_list, axis=0), axis=0)
df_importance = pd.concat(
    [
        pd.DataFrame(imporatance_ave, columns=['imporatance_ave']),
        pd.DataFrame(columns, columns=['feature_name'])
    ],
    axis=1
)
df_importance = df_importance.sort_values('imporatance_ave', ascending=False)

# feature_importance barplot
data_head = df_importance.head(100)
data_tail = df_importance[df_importance['imporatance_ave']==0]
palette = sns.diverging_palette(220, 20, n=len(data_head))
plt.figure(figsize=(6,20))
sns.barplot(x="imporatance_ave", y="feature_name", data=data_head, palette=palette) 

#save junk_features
path = '../feature/'
with open('{}xgb_junk_features.binaryfile'.format(path), 'wb') as f:
    pickle.dump(list(data_tail['feature_name']), f)
with open('{}xgb_top_features.binaryfile'.format(path), 'wb') as f:
    pickle.dump(list(data_head['feature_name']), f)