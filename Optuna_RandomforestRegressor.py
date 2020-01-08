from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import optuna

def obj_rf(trial):
    # parameter space
    max_depth=trial.suggest_int('max_depth', 100, 5000)
    max_features=trial.suggest_int('max_features', 10, X_train.shape[1])
    max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 100, 5000)
    max_samples=trial.suggest_int('max_samples', 50, X_train.shape[0]//2)
    min_impurity_decrease=trial.suggest_loguniform('min_impurity_decrease', 2**-7, 1)
    min_impurity_split=trial.suggest_loguniform('min_impurity_split', 2**-7, 1)
    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, X_train.shape[0]//2)
    min_samples_split=trial.suggest_int('min_samples_split', 2, X_train.shape[0]//2)
    min_weight_fraction_leaf=trial.suggest_uniform('min_weight_fraction_leaf', 0, 0.5)
    n_estimators=trial.suggest_int('n_estimators', 100, 2000)
    # cross validation
    scoring_list = []
    for train_index, valid_index in KFold(n_splits=5, random_state=0).split(X_train, y_train):
        # model training
        model = RandomForestRegressor(bootstrap=True,
                                      ccp_alpha=0.0,
                                      criterion='mse',
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      max_leaf_nodes=max_leaf_nodes,
                                      max_samples=max_samples,
                                      min_impurity_decrease=min_impurity_decrease,
                                      min_impurity_split=min_impurity_split,
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                      n_estimators=n_estimators,
                                      n_jobs=-1,
                                      oob_score=False,
                                      random_state=0,
                                      verbose=0,
                                      warm_start=False
                                     )
        model.fit(X_train[train_index], y_train[train_index])
        # scoring
        y_true = y_train[valid_index]
        y_pred = model.predict(X_train[valid_index])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scoring_list.append(rmse)
        
    return np.mean(scoring_list)

study = optuna.create_study()
study.optimize(obj_rf, n_trials=2, n_jobs=-1)