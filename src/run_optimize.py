import argparse
import pickle
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from processor import load_data
from optimizer import Objective, optuna_search

path_to_params = '../params/'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('modeltype', help='LGB, XGB, NN etc...')
    args = parser.parse_args()

    x, y = load_data()
    print('x_shape :', x.shape)
    print('y_shape :', y.shape)

    if args.modeltype == 'LGB':
        obj = Objective(LGBMRegressor(), x, y)
        OPTIMAL_PARAMAS = optuna_search(obj, 10, -1, 123)
        # args(obj_func, n_trials, n_jobs, seed)
        print(OPTIMAL_PARAMAS)
        with open('{}LGB_OPT_PARAMS.binaryfile'.format(path_to_params), 'wb') as f:
            pickle.dump(OPTIMAL_PARAMAS, f)

    elif args.modeltype == 'XGB':
        obj = Objective(XGBRegressor(), x, y)
        OPTIMAL_PARAMAS = optuna_search(obj, 10, -1, 123)
        # args(obj_func, n_trials, n_jobs, seed)
        print(OPTIMAL_PARAMAS)
        with open('{}XGB_OPT_PARAMS.binaryfile'.format(path_to_params), 'wb') as f:
            pickle.dump(OPTIMAL_PARAMAS, f)
