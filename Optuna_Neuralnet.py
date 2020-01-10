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

# load sample data
def boston_data():
    boston = load_boston()
    X = boston['data']
    y = boston['target']
    return X, y
X, y = boston_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.1)

class MLP:
    
    def __init__(self):
        self.input_dropout = 0.1
        self.hidden_dropout = 0.1
        self.n_layers = 2
        self.batch_norm = 'act'
        self.units = 9
        self.activation = 'relu'
        self.batch_size = 32
        self.model=None
        self.scaler = StandardScaler()
        
        
    def create_graph (self, input_dropout, hidden_dropout, n_layers, batch_norm, units, activation):
        self.model = Sequential()
        # input layer
        self.model.add(Dropout(self.input_dropout, input_shape=(X.shape[1],)))
        # hidden layer
        for i in range(self.n_layers):
            self.model.add(Dense(self.units))
            if self.batch_norm == 'act':
                self.model.add(BatchNormalization())
            if self.activation == 'prelu':
                self.model.add(PReLU())
            elif self.activation == 'relu':
                self.model.add(ReLU())
            self.model.add(Dropout(self.hidden_dropout))
            # output layer
        self.model.add(Dense(1))
        
    def fit(self, X_train, X_valid, y_train, y_valid, batch_size):
        # 標準化
        x_train = self.scaler.fit_transform(X_train)
        x_valid = self.scaler.fit_transform(X_valid)
        # optimizer
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.)
        # Compile
        history = self.model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mae']
        )
        # Batch learning and Early stopping
        early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
        self.model.fit(
            x_train, y_train,
            epochs=200,
            batch_size=self.batch_size, verbose=1,
            validation_data=(x_valid, y_valid),
            callbacks=[early_stopping]
        )
    
    def predict(self, X):
        #標準化
        x = self.scaler.fit_transform(X)
        y_pred = self.model.predict(x).flatten()
        return y_pred

def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(obj, n_trials=2) # *n_jobsを-1に設定するとエラー出る*

main()