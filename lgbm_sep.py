import multiprocessing
import os
from functools import partial

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold

from constants import IN_PATH, COLS_AUX, TARGET, COLS_USE, COLS_SEP_MODEL_DICT
from preprocess import Preprocessor


def train_one_fold(args, X, y, df):
    fd, train_index, test_index = args
    train_data = X.iloc[train_index]
    preprocessor = Preprocessor()
    train_data = preprocessor.process_features(train_data)
    test_data = X.iloc[test_index]
    test_df = df.iloc[test_index]
    preprocessor = Preprocessor()
    test_data = preprocessor.process_features(test_data)
    train_target, test_target = y[train_index], y[test_index]

    """train separate models on each gauge type"""
    for _, g in enumerate([[0], [1], [2, 3]]):
        if len(g) == 1:
            i = g[0]
        else:
            i = 2

        cols_model_i = COLS_SEP_MODEL_DICT[i]["cols"]
        train_data_i = train_data[cols_model_i]
        test_data_i = test_data[cols_model_i]
        tr_idx = train_data[train_data['Downhole Gauge Type'].isin(g)].index
        te_idx = test_data[test_data['Downhole Gauge Type'].isin(g)].index

        if g == [0]:
            mdl = LGBMRegressor(objective='mae', n_estimators=267, num_leaves=87, min_child_samples=41,
                                learning_rate=0.0425, colsample_bytree=0.478, reg_alpha=0.03, reg_lambda=0.03)
        elif g == [1]:
            mdl = LGBMRegressor(objective='mae', n_estimators=9, num_leaves=4, min_child_samples=3,
                                learning_rate=0.837, colsample_bytree=0.3, reg_alpha=4.2, reg_lambda=7.86)
        else:
            mdl = LGBMRegressor(objective='mae', n_estimators=12, num_leaves=7, min_child_samples=10,
                                learning_rate=0.565, colsample_bytree=0.05, reg_alpha=0.01, reg_lambda=2.39)

        mdl.fit(X=train_data_i.loc[tr_idx], y=train_target.loc[tr_idx],
                eval_set=[(train_data_i.loc[tr_idx], train_target.loc[tr_idx]),
                          (test_data_i.loc[te_idx], test_target.loc[te_idx])], verbose=50,
                categorical_feature=COLS_SEP_MODEL_DICT[i]["cats"])
        pred = mdl.predict(test_data_i.loc[te_idx])
        test_mae = mean_absolute_error(test_target.loc[te_idx], pred)
        print(f"test mae: {test_mae} for fold {fd} and g {g}")
        test_df.loc[te_idx, 'pred'] = pred

    return test_df[COLS_AUX + [TARGET, 'Downhole Gauge Type', 'pred']].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_feather(os.path.join(IN_PATH, "public"))

    X, y = df[COLS_USE], df[TARGET]
    well_number = df['Well_Number']
    operating_area_name = df['Operating_Area_Name']

    skf = StratifiedGroupKFold(n_splits=5)
    pool = multiprocessing.Pool(15)
    fold_preds = pool.map(partial(train_one_fold, X=X, y=y, df=df),
                          [(fd, train_index, test_index) for fd, (train_index, test_index) in
                           enumerate(skf.split(X, operating_area_name, well_number))])
    fold_preds = pd.concat(fold_preds)
    print(mean_absolute_error(fold_preds['Downhole Gauge Pressure'], fold_preds['pred']))
    # 432.2602009716844
