import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import keras
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
import optuna
from load_data import load_csv
from config import params


# params
params = params()
params_mlp = params['Regressor']['mlp']

# data sets
X_train, y_train, columns = load_csv()
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('len(len(columns)', len(columns))

# obj_bin
labels = np.arange(10)
y_train_bins = pd.cut(y_train, 10, labels=labels)


def create_gragh(input_dropout, hidden_dropout, n_layers, batch_norm, units):
    model = Sequential()
    # input layer
    model.add(Dropout(input_dropout, input_shape=(X_train.shape[1],)))
    # hidden layer
    for i in range(n_layers):
        model.add(Dense(units))
        if batch_norm == 'act':
            model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(hidden_dropout))
    # output layer
    model.add(Dense(1))
    # optimizer
    optimizer = Adam(lr=params_mlp['lr'], beta_1=0.9, beta_2=0.999, decay=0.)
    # Compile
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mae']
    )
    return model


def obj(trial):
    # clear session
    keras.backend.clear_session()
    # parameter space
    input_dropout = trial.suggest_uniform('input_dropout', 0, 0.3)
    hidden_dropout = trial.suggest_uniform('hidden_dropout', 0, 0.3)
    n_layers = trial.suggest_int('n_layers', 2, 4)
    units = int(trial.suggest_discrete_uniform('units', 32, 256, 32))
    batch_norm = trial.suggest_categorical('batch_norm', ['act', 'non'])
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 32, 128, 32))
    # create gragh
    model = create_gragh(
        input_dropout=input_dropout,
        hidden_dropout=hidden_dropout,
        n_layers=n_layers,
        batch_norm=batch_norm,
        units=units
        )
    # Early stopping
    early_stopping = EarlyStopping(
        patience=params_mlp['patience'], 
        restore_best_weights=True
        )
    # CV
    n_splits = params['Regressor']['cv_folds']
    random_state = params['Regressor']['cv_random_state']
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    rmse_list = []
    for tr_idx, va_idx in kf.split(X_train, y_train_bins):
        # StandardScaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X_train[tr_idx])
        x_valid = scaler.fit_transform(X_train[va_idx])
        model.fit(
            x_train, y_train[tr_idx],
            epochs=params_mlp['epochs'],
            batch_size=batch_size, verbose=0,
            validation_data=(x_valid, y_train[va_idx]),
            callbacks=[early_stopping]
        )
        # prediction
        y_true = y_train[va_idx]
        y_pred = model.predict(X_train[va_idx]).flatten()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


# optuna
n_trials = params['Regressor']['optuna_trials']
study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=1541)
)
study.optimize(obj, n_trials=n_trials, 
# n_jobs=-1
)

print('MLP_RMSE: ', study.best_value)
print('MLP_params: ', study.best_params)

# save
with open('../parameter/params_mlp.binaryfile', 'wb') as f:
    pickle.dump(study.best_params, f)

# # class
# class MLP:


#     def __init__(self):
#         self.input_dropout = 0.1
#         self.hidden_dropout = 0.1
#         self.units = 32
#         self.n_layers = 2
#         self.batch_norm = 'act'
#         self.batch_size = 32
#         self.model=None
#         self.scaler = StandardScaler()


#     def create_graph (self, input_dropout, hidden_dropout, n_layers, batch_norm, units):
#         keras.backend.clear_session()
#         self.model = Sequential()
#         # input layer
#         self.model.add(Dropout(self.input_dropout, input_shape=(X_train.shape[1],)))
#         # hidden layer
#         for i in range(self.n_layers):
#             self.model.add(Dense(self.units))
#             if self.batch_norm == 'act':
#                 self.model.add(BatchNormalization())
#             self.model.add(PReLU())
#             self.model.add(Dropout(self.hidden_dropout))
#             # output layer
#         self.model.add(Dense(1))


#     def fit(self, X_train, y_train, X_valid, y_valid, batch_size):
#         # StandardScaler
#         x_train = self.scaler.fit_transform(X_train)
#         x_valid = self.scaler.fit_transform(X_valid)
#         # optimizer
#         optimizer = Adam(lr=params_mlp['lr'], beta_1=0.9, beta_2=0.999, decay=0.)
#         # Compile
#         history = self.model.compile(
#             loss='mean_squared_error',
#             optimizer=optimizer,
#             metrics=['mae']
#         )
#         # Batch learning and Early stopping
#         early_stopping = EarlyStopping(
#             patience=params_mlp['patience'], 
#             restore_best_weights=True
#             )
#         self.history = self.model.fit(
#             x_train, y_train,
#             epochs=params_mlp['epochs'],
#             batch_size=self.batch_size, verbose=0,
#             validation_data=(x_valid, y_valid),
#             callbacks=[early_stopping]
#         )
#         return self.history

#     def predict(self, X):
#         # StandardScaler
#         x = self.scaler.fit_transform(X)
#         y_pred = self.model.predict(x).flatten()
#         return y_pred

# #######################################################



# # Objective function
# def obj(trial):
#     # parameter space
#     input_dropout = trial.suggest_uniform('input_dropout', 0, 0.3)
#     hidden_dropout = trial.suggest_uniform('hidden_dropout', 0, 0.3)
#     n_layers = trial.suggest_int('n_layers', 2, 4)
#     units = int(trial.suggest_discrete_uniform('units', 32, 256, 32))
#     batch_norm = trial.suggest_categorical('batch_norm', ['act', 'non'])
#     batch_size = int(trial.suggest_discrete_uniform('batch_size', 32, 128, 32))
#     # model building
#     model = MLP()
#     model.create_graph(
#         input_dropout=input_dropout,
#         hidden_dropout=hidden_dropout,
#         n_layers=n_layers,
#         batch_norm=batch_norm,
#         units=units
#     )
#     # CV
#     n_splits = params['Regressor']['cv_folds']
#     random_state = params['Regressor']['cv_random_state']
#     kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

#     rmse_list = []
#     for tr_idx, va_idx in kf.split(X_train, y_train_bins):
#         model.fit(
#             X_train[tr_idx], y_train[tr_idx],
#             X_train[va_idx], y_train[va_idx],
#             batch_size
#         )
#         # prediction
#         y_true = y_train[va_idx]
#         y_pred = model.predict(X_train[va_idx]).flatten()
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         rmse_list.append(rmse)
#     return np.mean(rmse_list)

