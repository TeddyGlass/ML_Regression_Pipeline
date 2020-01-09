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

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1217, test_size=0.1)

st = StandardScaler()
x_tr = st.fit_transform(X_train)
x_va = st.fit_transform(X_test)

model = Sequential()
model.add(Dropout(rate=0.3, input_shape=(x_tr.shape[1],)))
model.add(Dense(1))

optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.)

model.compile(loss='mean_squared_error',
                           optimizer=optimizer, metrics=['accuracy'])
nb_epoch = 200
patience = 20
early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
model.fit(x_tr, y_train,
          epochs=nb_epoch,
          batch_size=32, verbose=1,
          validation_data=(x_va, y_test),
          callbacks=[early_stopping])