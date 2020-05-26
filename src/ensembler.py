from trainer import Trainer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class Ensembler:

    def __init__(self, model):
        self.model = model
        self.early_stopping_rounds = 1000
        self.n_splits = 10
        self.random_state = 2031
        self.kf = KFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=True
            )
        self.va_idxes = []
        self.va_preds = []
        self.te_preds = []
        self.rmse_list = []
        self.r2_list = []
        self.model_list = []

    def ensemble(self, x_train, y_train, x_test, y_test):
        for tr_idx, va_idx in self.kf.split(x_train, y_train):
            self.reg = Trainer(self.model)
            self.reg.fit(
                x_train[tr_idx],
                y_train[tr_idx],
                x_train[va_idx],
                y_train[va_idx],
                self.early_stopping_rounds
            )
            # prediction
            va_pred = self.reg.predict(x_train[va_idx])
            te_pred = self.reg.predict(x_test)
            # evaluation
            va_true = y_train[va_idx]
            rmse = np.sqrt(mean_squared_error(va_true, va_pred))
            r2 = r2_score(va_true, va_pred)
            # list up
            self.rmse_list.append(rmse)
            self.r2_list.append(r2)
            self.model_list.append(self.reg.get_model())
            self.va_idxes.append(va_idx)
            self.va_preds.append(va_pred)
            self.te_preds.append(te_pred)

    def get_prediction_feature(self):
        # sort and processing prediction instance
        valid_index = np.concatenate(self.va_idxes)
        order = np.argsort(valid_index)
        train_preds = np.concatenate(self.va_preds)[order]
        test_preds = np.mean(self.te_preds, axis=0)
        return train_preds, test_preds

    def get_rmse(self):
        rmse_mean = np.mean(self.rmse_list)
        rmse_var = np.var(self.rmse_list)
        cv = 100*(rmse_mean/rmse_var)
        print('each_RMSE :', self.rmse_list)
        print('mean_RMSE :', rmse_mean)
        print('RMSE CV(%) :', cv)
        return self.rmse_list, rmse_mean, cv

    def get_r2(self):
        r2_mean = np.mean(self.r2_list)
        r2_var = np.var(self.r2_list)
        cv = 100*(r2_mean/r2_var)
        print('each_R2 :', self.r2_list)
        print('mean_R2 :', r2_mean)
        print('R2 CV(%) :', cv)
        return self.r2_list, r2_mean, cv

    def get_learning_curve(self):
        return self.reg.get_learning_curve()


if __name__ == "__main__":
    pass
