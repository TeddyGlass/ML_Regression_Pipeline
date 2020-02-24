
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold


class Trainer:

    def __init__(self, model):
        self.model = model
        self.best_iteration = 100
        self.train_rmse = []
        self.valid_rmse = []
        self.importance = []

    def fit(self,
            X_train, y_train, X_valid, y_valid,
            early_stopping_rounds):

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        if type(self.model).__name__ == "LGBMRegressor":
            self.model. fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric='root_mean_squared_error',
                verbose=False
            )
            self.best_iteration = self.model.best_iteration_
            self.train_rmse = np.array(
                self.model.evals_result_['valid_0']['rmse'])
            self.valid_rmse = np.array(
                self.model.evals_result_['valid_1']['rmse'])
            self.importance = self.model.booster_.feature_importance(
                importance_type='gain')

        elif type(self.model).__name__ == 'XGBRegressor':
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric='rmse',
                verbose=0
            )
            self.best_iteration = self.model.best_iteration
            self.train_rmse = np.array(
                self.model.evals_result_['validation_0']['rmse'])
            self.valid_rmse = np.array(
                self.model.evals_result_['validation_1']['rmse'])
            self.importance = self.model.feature_importances_

        # elif type(self.model).__name__ == 'CatBoostRegressor':
        #     self.model.fit(
        #         X_train,
        #         y_train,
        #         early_stopping_rounds=early_stopping_rounds,
        #         eval_set=eval_set,
        #         use_best_model=True,
        #         verbose=False
        #     )
        #     self.best_iteration = self.model.get_best_iteration()

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_importance(self):
        return self.importance

    def get_lerning_curve(self):
        palette = sns.diverging_palette(220, 20, n=2)
        width = np.arange(self.train_rmse.shape[0])
        plt.figure(figsize=(10, 7.32))
        plt.title('Learning_Curve', fontsize=15)
        plt.xlabel('Estimators', fontsize=15)
        plt.ylabel('RMSE', fontsize=15)
        plt.plot(width, self.train_rmse, label='train_ramse', color=palette[0])
        plt.plot(width, self.valid_rmse, label='valid_rmse', color=palette[1])
        plt.legend(loc='upper right', fontsize=13)
        plt.show()

    def predict(self, X):
        if type(self.model).__name__ == "LGBMRegressor":
            self.model.predict(X, ntree_limit=self.best_iteration)

        elif type(self.model).__name__ == 'CatBoostRegressor':
            self.model.predict(X, ntree_end=self.best_iteration)

        elif type(self.model).__name__ == 'XGBRegressor':
            self.model.predict(X, ntree_limit=self.best_iteration)


# class CV_training:

#     def __init__(self, model):
#         super(Trainer, self).__init__(model)

#     def cv_training(self, X, y, n_splits, random_state, kf_type):
#         if kf_type == 'kf':
#             kf = KFold(n_splits=n_splits,
#                        random_state=random_state, shuffle=True)
#         elif kf_type == 'skf':
#             kf = StratifiedKFold(
#                 n_splits=n_splits, random_state=random_state, shuffle=True)

#         for tr_idx, va_idx in kf.split(X, y):
#             super().fit(
#                 X[tr_idx],
#                 y[tr_idx],
#                 X[va_idx],
#                 y[va_idx],
#                 early_stopping_rounds
#             )
