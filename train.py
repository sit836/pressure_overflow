"""Unearthed Training Template"""
import argparse
import logging
import pickle
import sys
from io import StringIO
from os import getenv
from os.path import abspath, join

import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping
from sklearn.model_selection import StratifiedKFold

from constants import TARGET, COLS_USE
from ensemble_model import EnsembleModel
from preprocess import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # If you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = pd.read_csv(join(args.data_dir, "public.csv.gz"), parse_dates=True)
    X, y = df[COLS_USE], df[TARGET]

    preprocessor = Preprocessor()
    X_processed = preprocessor.process_features(X)

    skf = StratifiedKFold(n_splits=5)
    models = []
    for fd, (train_index, test_index) in enumerate(skf.split(X_processed, X_processed['Operating_Area_Name'])):
        train_data = X_processed.iloc[train_index]
        test_data = X_processed.iloc[test_index]
        train_target, test_target = y[train_index], y[test_index]
        mdl = lgb.LGBMRegressor(objective='mae', n_estimators=7018, learning_rate=0.005774021709350338,
                                min_child_samples=19, num_leaves=69, reg_alpha=750.4226963536518,
                                reg_lambda=5.903826059674324)

        early_stop_callback = early_stopping(25)
        mdl.fit(X=train_data, y=train_target,
                eval_set=[(train_data, train_target),
                          (test_data, test_target)],
                callbacks=[early_stop_callback], verbose=50)
        models.append(mdl)

    # save the model to disk
    save_model(EnsembleModel(models), args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)
    logger.info(f"model saved to {model_dir}")


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info("receiving preprocessed input")

    # this call must result in a dataframe or nparray that matches your model
    input = pd.read_csv(StringIO(input_data), index_col=0, parse_dates=True)
    logger.info(f"preprocessed input has shape {input.shape}")
    return input


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.

    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )

    train(parser.parse_args())
