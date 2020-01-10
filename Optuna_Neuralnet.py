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

# data sets
boston = load_boston()
X = boston['data']
y = boston['target']

def obj_nn(trial):
    input_dropout = trial.suggest_uniform('input_dropout', 0, 0.3)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 5)
    hidden_units = trial.suggest_int('hidden_units', 6,26)
    batch_norm = trial.suggest_categorical('batch_norm', ['before_act', 'no'])
    hidden_activation = trial.suggest_categorical('hidden_activation', ['prelu', 'relu'])
    hidden_dropout = trial.suggest_uniform('hidden_dropout', 0.0, 0.3)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    
    model = Sequential()
    # input layer
    model.add(Dropout(input_dropout, input_shape=(X.shape[1],)))
    
    # hidden layer
    for i in range(hidden_layers):
        model.add(Dense(hidden_units))
        if batch_norm == 'before_act':
            model.add(BatchNormalization())
        if hidden_activation == 'prelu':
            model.add(PReLU())
        elif hidden_activation == 'relu':
            model.add(ReLU())
        else:
            raise NotImplementedError
        model.add(Dropout(hidden_dropout))
    
    # Output layer
    model.add(Dense(1))
    
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-3,1e-2)
    optimizer = Adam(lr=adam_lr, beta_1=0.9, beta_2=0.999, decay=0.)
    
    # Evaluatioon
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mae']
    )

    
    # Epoch and Early stopping
    nb_epoch = 200
    patience = 20
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

    # traininig
    scoring_list = []
    for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
        x_tr = scaler.fit_transform(X_train[train_index])
        y_tr = y_train[train_index]
        x_va = scaler.fit_transform(X_train[valid_index])
        y_va = y_train[valid_index]
        history = model.fit(
            x_tr, y_tr,
            epochs=nb_epoch,
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

# try stydy
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.1)
study = optuna.create_study()
study.optimize(obj_nn, n_trials=2, n_jobs=-1)