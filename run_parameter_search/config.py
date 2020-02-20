def params():
    param = {
        'Regressor': {
            'cv_folds': 5,
            'cv_random_state': 4321,
            'optuna_trials': 60,
            'lightgbm': {
                'learning_rate': 0.1,
                'n_estimators': 10000,
                'max_depth': -1,
                'num_leaves': 31,
                'subsample': 0.65,
                'colsample_bytree': 0.65,
                'min_child_weight': 10,
                'min_split_gain': 0.01,
                'random_state': 1112,
                'n_jobs': -1
            },
            'xgboost': {
                'learning_rate': 0.1,
                'n_estimators': 10000,
                'max_depth': 9,
                'subsample': 0.65,
                'colsample_bytree': 0.65,
                'gamma': 1,
                'min_child_weight': 10,
                'random_state': 1112,
                'n_jobs': -1
            },
            'catboost': {
                'iterations': 10000,
                'depth': 7,
                'learning_rate': 0.1,
                'random_strength': 1,
                'bagging_temperature': 1,
                'grow_policy': 'SymmetricTree',
                'eval_metric': 'RMSE',
                'random_state': 1112
            },
            'mlp': {
                'epochs': 10000,
                'patience': 50,
                'lr': 0.05
                },
            'randomforest': {
                'n_estimators': 300,
                'max_depth': 10,
                # 'max_leaf_nodes': 1
                # leafの最大数 depthを大きくするとかなりメモリを食う
                'max_features': 10,  # 分割feature探索に用いるfeatureの数
                'max_samples': 10,  # ブートストラップ時のサンプリング数
                'min_impurity_decrease': 1e-2,
                'min_samples_split': 10,
                'n_jobs': -1,
                'random_state': 1112
            }
        }
    }
    return param
