import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
import optuna
import pandas as pd

# load data
def boston_data():
    boston = load_boston()
    X = boston['data']
    y = boston['target']
    return X, y
X, y = boston_data()

# model building
def create_model(input_dropout, hidden_dropout, n_layers, batch_norm, units, activation):
    model = Sequential()
    
    # input layer
    model.add(Dropout(input_dropout, input_shape=(X.shape[1],)))
    
    # hidden layer
    for i in range(n_layers):
        model.add(Dense(units))
        if batch_norm == 'act':
            model.add(BatchNormalization())
        if activation == 'prelu':
            model.add(PReLU())
        if activation == 'relu':
            model.add(ReLU())
    model.add(Dropout(hidden_dropout))
            
    # output layer
    model.add(Dense(1))
    
    return model

# Objective function
def obj(trial):
# parameter space
input_dropout = trial.suggest_uniform('input_dropout', 0, 0.3)
hidden_dropout = trial.suggest_uniform('hidden_dropout', 0, 0.3)
n_layers = trial.suggest_int('n_layers', 1, 5)
units = int(trial.suggest_discrete_uniform('units', 6, 26, 1))
batch_norm = trial.suggest_categorical('batch_norm', ['act', 'no'])
activation = trial.suggest_categorical('activation', ['prelu', 'relu'])
batch_size = int(trial.suggest_discrete_uniform('batch_size', 32, 128, 32))
# model building
model = create_model(
    input_dropout=input_dropout,
    hidden_dropout=hidden_dropout,
    n_layers=n_layers,
    units=units,
    activation=activation,
    batch_norm=batch_norm
)
# optimizer
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.)
# Compile
model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['mae']
)
# Epoch and Early stopping
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
# train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.1)
scoring_list = []
for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(X_train[train_index])
    x_va = scaler.fit_transform(X_train[valid_index])
    y_tr = y_train[train_index]
    y_va = y_train[valid_index]
    history = model.fit(
        x_tr, y_tr,
        epochs=200,
        batch_size=batch_size, verbose=1,
        validation_data=(x_va, y_va),
        callbacks=[early_stopping]
    )
    # prediction
    y_true = y_va
    y_pred = model.predict(x_va).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    scoring_list.append(rmse)
    
return np.mean(scoring_list)  


def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(obj, n_trials=50) # *n_jobsを-1に設定するとエラー出る*
    print(study.best_params)
    return study