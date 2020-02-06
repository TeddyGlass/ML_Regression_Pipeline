def params():
    param = {
        'Regressor': {
            'cv_folds': 2,
            'cv_random_state': 1234,
            'optuna_trials': 10,
            'lightgbm': {
                'learning_rate': 0.05,
                'n_estimators': 10000,
                'max_depth': 7,
                'num_leaves': int(0.8*(2**7)),
                'subsample': 0.65,
                'col_sample_bytree': 0.65,
                'min_child_weight': 10,
                'min_split_gain': 0.01,
                'random_state': 1112,
                'n_jpbs': -1
            },
            'xgboost': {
                'learning_rate': 0.05,
                'n_estimators': 10000,
                'max_depth': 7,
                'subsample': 0.65,
                'colsample_bytree': 0.65,
                'gamma': 1,
                'min_child_weight': 10,
                'random_state': 1112,
                'n_jobs': -1
            }
        }
    }
    return param
