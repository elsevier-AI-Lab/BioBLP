import torch
import string
import numpy as np
import random as rn

from pathlib import Path


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


from sklearn.model_selection import train_test_split

from typing import Union, Tuple

from bioblp.logger import get_logger


logger = get_logger(__name__)


def get_random_string(length):
    # choose from all lowercase letter
    characters = string.ascii_lowercase + string.digits
    result_str = "".join(rn.choice(characters) for i in range(length))

    return result_str


def unique_study_prefix():
    unique_string = get_random_string(8)
    return unique_string


def generate_study_name(prefix, model, fold):
    return f"{prefix}-{model}-{fold}"


def aupr_score(y_true, y_pred):
    """Use AUC function to calculate the area under the curve of precision recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def get_auc_scorers():
    scorers = {
        "PRCURVE": make_scorer(precision_recall_curve, needs_proba=True),
        "ROCCURVE": make_scorer(roc_curve, needs_proba=True)
    }
    return scorers


def get_scorers():
    scorers = {
        "AUCROC": make_scorer(roc_auc_score, needs_proba=True),
        "f1": make_scorer(fbeta_score, beta=1, average="micro"),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "accuracy": make_scorer(accuracy_score),
        "AUCPR": make_scorer(aupr_score, needs_proba=True),
    }
    return scorers


def get_model_label(feature: str, model: str):
    return f"{feature}__{model}"


def load_feature_data(feat_path: Union[str, Path], dev_run: bool = False) -> Tuple[np.array, np.array]:
    """ Load feature data into numpy arrays

    Parameters
    ----------
    feat_path : Union[str, Path]
        Filepath to feature, eg 'features/rotate.pt'
    dev_run : bool, optional
        Flag to subsample data for development only, by default False

    Returns
    -------
    Tuple[np.array, np.array]
        Return (features, labels)
    """
    logger.info("Loading training data...")

    data = torch.load(feat_path)

    X = data.get("X")
    y = data.get("y")

    if torch.is_tensor(X):
        X = X.detach().numpy()
        y = y.detach().numpy()

    if dev_run:
        X, _, y, _ = train_test_split(
            X, y, stratify=y, train_size=0.1, random_state=12)

    logger.info(
        "Resulting shapes X: {}, y: {}".format(
            X.shape, y.shape)
    )
    logger.info("Counts in y: {}".format(
        np.unique(y, return_counts=True)))

    return X, y


def validate_features_exist(feature_dir: Path, models_conf: dict) -> bool:
    """ Check if all feature files exist in directory

    Parameters
    ----------
    feature_dir : Path
        Path to feature location
    models_conf : dict
        Definition of model and feature.

    Returns
    -------
    bool
        True if features are present.
    """
    exists = {}

    all_features = list(set([v.get("feature")
                             for _, v in models_conf.items()]))

    for feat in all_features:
        exists[feat] = feature_dir.joinpath(f"{feat}.pt").is_file()

    logger.info(f"Validated that features exist: {exists}..")

    missing = [k for k, v in exists.items() if v is False]
    if len(missing) > 0:
        logger.warning(f"Missing features {missing}!!")

    return all([v for _, v in exists.items()])
