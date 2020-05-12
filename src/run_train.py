import argparse
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from processor import load_data, load_test
from ensembler import Ensembler
from conf import Paramset
from utils import load, save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('modeltype', help='LGB, XGB, NN etc...')
    args = parser.parse_args()

    # setting train cv parameter
    n_splits = 5
    random_state = 2004
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # path
    ex = '.binaryfile'
    path_to_params = '../params/{}_OPT_PARAMS{}'.format(args.modeltype, ex)

    # load data
    x_train, y_train = load_data()
    x_test, y_test = load_test()
    print('x_shape :', x_train.shape)
    print('y_shape :', y_train.shape)

    # load optimized parameters
    OPT_PARAMS = load(path_to_params)

    if args.modeltype == 'LGB':
        # generate model base parameters and update parameters
        paramset = Paramset(LGBMRegressor())
        paramset.swiching_lr('train')
        LGB_PARAMS = paramset.generate_params()
        LGB_PARAMS.update(OPT_PARAMS)
        LGB_PARAMS.update(
            min_child_samples=int(OPT_PARAMS['min_child_samples']),
            bagging_freq=int(OPT_PARAMS['bagging_freq'])
        )
        print(LGB_PARAMS)
        # ensemble
        esm = Ensembler(LGBMRegressor(), LGB_PARAMS)
        esm.ensemble(
            x_train,
            y_train,
            x_test,
            y_test
            )
        # evaluation
        rmse_each, rmse_mean, rmse_cv = esm.get_rmse()
        r2_each, r2_mean, r2_cv = esm.get_r2()
        esm.get_learning_curve()
        # save prediction fearures
        train_preds, test_preds = esm.get_prediction_feature()
        ITEMS = [train_preds, test_preds]
        NAMES = ['LGB_preds_train', 'LGB_preds_test']
        for item, name in zip(ITEMS, NAMES):
            save('../preds/{}{}'.format(name, ex), item)
  
    elif args.modeltype == 'XGB':
        # generate model base parameters and update parameters
        paramset = Paramset(XGBRegressor())
        paramset.swiching_lr('train')
        XGB_PARAMS = paramset.generate_params()
        XGB_PARAMS.update(OPT_PARAMS)
        # ensemble
        esm = Ensembler(XGBRegressor(XGB_PARAMS))
        esm.ensemble(
            x_train,
            y_train,
            x_test,
            y_test
            )
        # evaluation
        rmse_each, rmse_mean, rmse_cv = esm.get_rmse()
        r2_each, r2_mean, r2_cv = esm.get_r2()
        esm.get_learning_curve()
        # save prediction fearures
        train_preds, test_preds = esm.get_prediction_feature()
        ITEMS = [train_preds, test_preds]
        NAMES = ['XGB_preds_train', 'XGB_preds_test']
        for item, name in zip(ITEMS, NAMES):
            save('../preds/{}{}'.format(name, ex), item)