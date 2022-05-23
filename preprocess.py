import argparse
import logging

import pandas as pd

from constants import COLS_NUM_TO_ENC, WELL_NUMBER, ELAPSED_DAYS, DATE, \
    COLS_AUX, CAT_FEA_DICT, COLS_NUM_NO_ENC, COLS_CAT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COLS_USE = COLS_NUM_TO_ENC + COLS_NUM_NO_ENC + COLS_CAT + COLS_AUX


class Preprocessor:
    def __init__(self):
        self.feature_names = None
        self.is_geo_clusters_fitted = False

    def _store_var_names(self, df):
        self.feature_names = df.columns

    def _encode_cat(self, df):
        for cat_fea, _ in CAT_FEA_DICT.items():
            unknown_to_value = len(CAT_FEA_DICT[cat_fea])
            df[cat_fea] = df[cat_fea].map(lambda x: CAT_FEA_DICT[cat_fea].get(x, unknown_to_value))

        return df

    def _fill_missing(self, df, aux_df):
        return df.groupby(aux_df[WELL_NUMBER], sort=False).fillna(method="ffill").fillna(method="bfill")

    def _create_features(self, X_df, aux_df):
        def add_date_components(X_df, aux_df):
            aux_df[DATE] = pd.to_datetime(aux_df[DATE])
            X_df[ELAPSED_DAYS] = aux_df[DATE].groupby(aux_df[WELL_NUMBER], sort=False).transform(
                lambda x: (x - x.min()).dt.days)
            X_df['year'] = aux_df[DATE].dt.year
            X_df['month'] = aux_df[DATE].dt.month
            X_df['day'] = aux_df[DATE].dt.day
            X_df['day_of_week'] = aux_df[DATE].dt.dayofweek

            return X_df

        def add_rolling_statistics(X_df, aux_df):
            def add_rolling_statistics_by_well(X_df, aux_df):
                rolling_stat = []
                for window in window_sizes:
                    ma_df = X_df[COLS_NUM_TO_ENC].groupby(aux_df[WELL_NUMBER], sort=False).transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()).add_suffix(
                        f'_MA_{window}_days')
                    iqr_df = X_df[COLS_NUM_TO_ENC].groupby(aux_df[WELL_NUMBER], sort=False).transform(
                        lambda x: x.rolling(window=window, min_periods=1).quantile(third_quartile_lvl) - x.rolling(
                            window=window, min_periods=1).quantile(first_quartile_lvl)).add_suffix(
                        f'_IQR_{window}_days')

                    diff_ma_df = pd.DataFrame(X_df[COLS_NUM_TO_ENC].values - ma_df.values,
                                              columns=X_df[COLS_NUM_TO_ENC].columns + f"_diff_MA_{window}_days",
                                              index=X_df.index)
                    diff_iqr_df = pd.DataFrame(X_df[COLS_NUM_TO_ENC].values - iqr_df.values,
                                               columns=X_df[COLS_NUM_TO_ENC].columns + f"_diff_IQR_{window}_days",
                                               index=X_df.index)
                    rolling_stat.extend([ma_df, iqr_df, diff_ma_df, diff_iqr_df])

                return pd.concat([X_df, pd.concat(rolling_stat, axis=1)], axis=1)

            first_quartile_lvl, third_quartile_lvl = 0.25, 0.75
            window_sizes = [3, 7, 14, 30]
            X_df = add_rolling_statistics_by_well(X_df, aux_df)

            return X_df

        def add_ranges(X_df):

            X_df['Pump Speed Actual Range'] = X_df['Pump Speed Actual Max'] - X_df['Pump Speed Actual Min']
            X_df['Pump Speed Max2Actual'] = X_df['Pump Speed Actual Max'] - X_df['Pump Speed Actual']
            X_df['Pump Speed Min2Actual'] = X_df['Pump Speed Actual Min'] - X_df['Pump Speed Actual']

            return X_df

        def add_geo_features(X_df):

            X_df['Euc_Squared_Dist'] = X_df['Well_Surface_Latitude'] ** 2 + X_df['Well_Surface_Longitude'] ** 2

            return X_df

        def add_superposition(X_df):
            X_df['flowrate_superposition'] = X_df['Tubing Flow Meter'] + X_df['Water Flow Mag from Separator']
            X_df['pressure_superposition'] = X_df['Casing Pressure'] + X_df['Tubing Pressure'] + X_df[
                'Gas Gathering Pressure'] + X_df['Separator Gas Pressure']

            return X_df

        def add_boolean_features(X_df):
            col_names = ['Tubing Flow Meter', 'Water Flow Mag from Separator', 'FCV Position Feedback',
                         'Pump Speed Actual', 'Pump Speed Actual Min',
                         'Pump Speed Actual Max', 'Pump Torque']

            for col in col_names:
                X_df[f'is_{col}_zero'] = (X_df[col] == 0.0)
                X_df[f'is_{col}_neg'] = (X_df[col] < 0.0)

            return X_df

        X_df_processed = X_df
        X_df_processed = add_date_components(X_df_processed, aux_df)
        X_df_processed = add_rolling_statistics(X_df_processed, aux_df)
        X_df_processed = add_ranges(X_df_processed)
        X_df_processed = add_geo_features(X_df_processed)
        X_df_processed = add_superposition(X_df_processed)
        X_df_processed = add_boolean_features(X_df_processed)

        return X_df_processed

    def process_features(self, df):
        X_df, aux_df = df.drop(columns=COLS_AUX), df[COLS_AUX]

        if self.feature_names is None:
            self._store_var_names(X_df)
        X_df = X_df[self.feature_names]  # in case the order of features changes

        X_df_processed = X_df
        X_df_processed = self._fill_missing(X_df_processed, aux_df)
        X_df_processed = self._encode_cat(X_df_processed)
        X_df_processed = self._create_features(X_df_processed, aux_df)

        return X_df_processed


if __name__ == "__main__":
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=str, default="D:/py_projects/pressure_overflow/data/public.csv.gz")

    parser.add_argument(
        "--output",
        type=str,
        default="D:/py_projects/pressure_overflow/output/public.csv",
    )
    args, _ = parser.parse_known_args()

    df = pd.read_csv(args.input, parse_dates=True)
    X = df[COLS_USE]

    preprocessor = Preprocessor()
    X_processed = preprocessor.process_features(X)

    logger.info(f"df.shape: {df.shape}")
    logger.info(f"X_processed.shape: {X_processed.shape}")

    # write to the output location
    X_processed.to_csv(args.output)
