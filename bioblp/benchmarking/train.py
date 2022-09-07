import torch
import os
import numpy as np
import random as rn
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from time import time
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate


from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.logging import get_logger


logger = get_logger(__name__)


SEED = 2022
DATA_DIR = Path("/home/jovyan/BioBLP/data/")
DATA_SHARED = Path("/home/jovyan/workbench-shared-folder/bioblp")

#
# DEFAULT PARAMETERS FOR RR
#
rf_default_params = {
    "n_estimators": 300,
    "criterion": "gini",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}

lr_default_params = {
    "C": 1.0,
    "random_state": SEED,
    "max_iter": 1000,
    "solver": "lbfgs",
    "n_jobs": -1,
}


#
# OPT SPACES FOR RR
#

rf_search_space = {
    "criterion": ["gini", "entropy"],
    "n_estimators": np.arange(100, 300, 50, dtype=int),
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10, 15, 100],
    "max_depth": [5, 8, 15, 25, 30, None],
    "random_state": [SEED],
}

lr_search_space = {
    "penalty": ["l2"],
    "C": np.logspace(-4, 3, 8),
    "random_state": [SEED],
    "max_iter": [1000],
    "solver": ["lbfgs"],
    "n_jobs": [-1],
}


def run_nested_cv(
    candidates: list,
    X,
    y,
    scoring: dict,
    outer_n_folds: int = 5,
    inner_n_folds: int = 2,
    inner_n_iter: int = 10,
    shuffle: bool = False,
    random_state: int = SEED,
    n_jobs: int = 1,
    refit_param: str = "fbeta",
    verbose: int = 10,
) -> dict:
    """Nested cross validation routine.
    Inner cv loop performs hp optimization on all folds and surfaces
    Parameters
    ----------
    candidates : list
        list of (label, estimator, param_dist)
    X : np.array
        predictor
    y : np.ndarray
        labels
    scoring : dict
        dict containing sklearn scorers
    outer_n_folds : int, optional
        splits for outer cv, by default 5
    inner_n_folds : int, optional
        splits for inner cv, by default 2
    inner_n_iter : int, optional
        number of trials within inner fold, by default 10
    shuffle : bool, optional
        shuffles data before cv, by default True
    random_state : int, optional
        seed for rng, by default SEED
    n_jobs : int, optional
        multiprocessing, by default 10
    refit_param : str, optional
        which metric to optimize for and return refit model, by default 'fbeta'
    verbose : int, optional
        level of console feedback, by default 0
    Returns
    -------
    dict
        outer cv scores e.g. {name: scores}
    """
    gridcvs = {}

    inner_cv = StratifiedKFold(
        n_splits=inner_n_folds, shuffle=shuffle, random_state=random_state
    )

    for name, estimator, param_grid in candidates:
        gcv = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=inner_n_iter,
            scoring=scoring,
            n_jobs=1,
            cv=inner_cv,
            verbose=10,
            refit=refit_param,
            # random_state=random_state,
        )
        gridcvs[name] = gcv

    outer_cv = StratifiedKFold(
        n_splits=outer_n_folds, shuffle=shuffle, random_state=random_state
    )
    outer_scores = {}

    for name, gs_est in sorted(gridcvs.items()):
        logger.info(f"Running CV for {name}..")
        nested_score = cross_validate(
            gs_est,
            X=X,
            y=y,
            scoring=scoring,
            cv=outer_cv,
            n_jobs=n_jobs,
            return_estimator=False,
            return_train_score=True,
            verbose=10,
        )

        score_to_optimize = nested_score.get("test_{}".format(refit_param))
        logger.info(
            f"{name}: outer {refit_param} {100*score_to_optimize.mean():.2f} +/- {100*score_to_optimize.std():.2f}"
        )
        outer_scores[name] = nested_score
    return outer_scores


def aupr_score(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # Use AUC function to calculate the area under the curve of precision recall curve
    return auc(recall, precision)


def run(args):
    """Perform train run"""

    # reproducibility
    SEED = 2022

    # def set_seeds(seed: int = SEED):
    #     os.environ['PYTHONHASHSEED'] = str(SEED)
    #     np.random.seed(SEED)
    #     tf.random.set_seed(SEED)
    #     rn.seed(SEED)

    experiment_config = {
        "n_proc": args.n_proc,
        "n_iter": args.n_iter,
        "inner_n_folds": args.inner_n_folds,
        "outer_n_folds": args.outer_n_folds,
        "param": args.param,
    }

    # data_dir = experiment_base_path.joinpath(experiment_config["data_dir"])
    out_dir = Path(args.outdir)

    n_proc = experiment_config["n_proc"]
    n_iter = experiment_config["n_iter"]
    inner_n_folds = experiment_config["inner_n_folds"]
    outer_n_folds = experiment_config["outer_n_folds"]
    optimize_param = experiment_config["param"]

    # set_seeds(seed=SEED)

    shuffle = True

    exp_output = defaultdict(dict)
    exp_output["settings"] = {
        # 'data_dir': data_dir,
        "n_iter": n_iter,
        "inner_n_folds": inner_n_folds,
        "outer_n_folds": outer_n_folds,
        "optimize_param": optimize_param,
        "shuffle": shuffle,
        "seed": SEED,
    }

    start = time()

    logger.info("Starting model building script at {}.".format(start))

    ############
    # Load data
    ############
    logger.info("Loading training data...")

    X_train = np.load(DATA_DIR.joinpath("features/dpi_X.npy"))
    y_train = np.load(DATA_DIR.joinpath("features/dpi_y.npy"))

    logger.info(
        "Resulting shapes X_train: {}, y_train: {}".format(X_train.shape, y_train.shape)
    )
    logger.info("Counts in y_train: {}".format(np.unique(y_train, return_counts=True)))
    ############
    # Setup classifiers & pipelines
    ############

    lr_label = "LR"
    clf_lr = LogisticRegression(**lr_default_params)

    rf_label = "RF"
    clf_rf = RandomForestClassifier(**rf_default_params)

    # record default params
    exp_output["default_params"]: {
        lr_label: lr_default_params,
        rf_label: rf_default_params,
    }

    ############
    # Setup grids
    ############
    exp_output["grids"]: {lr_label: lr_search_space, rf_label: rf_search_space}

    ############
    # Compare models
    ############
    candidates = [
        (lr_label, clf_lr, lr_search_space),
        (rf_label, clf_rf, rf_search_space),
    ]

    scorer = {
        "AUROC": make_scorer(roc_auc_score, needs_proba=True),
        "fbeta": make_scorer(fbeta_score, beta=1, average="micro"),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "accuracy": make_scorer(accuracy_score),
        "AUPRC": make_scorer(aupr_score, needs_proba=True),
    }

    nested_cv_scores = run_nested_cv(
        candidates=candidates,
        X=X_train,
        y=y_train,
        scoring=scorer,
        inner_n_folds=inner_n_folds,
        inner_n_iter=n_iter,
        outer_n_folds=outer_n_folds,
        shuffle=shuffle,
        n_jobs=n_proc,
        refit_param=optimize_param,
        random_state=SEED,
    )

    for algo, scores in nested_cv_scores.items():
        logger.info("Scores {}: {}".format(algo, scores))

    exp_output["results"] = nested_cv_scores

    logger.info(exp_output)

    run_timestamp = int(time())
    file_out = out_dir.joinpath("nested_cv_scores_{}.npy".format(run_timestamp))
    logger.info("Saving to {}".format(file_out))
    np.save(file_out, exp_output)

    end = time()

    logger.info("Ran script in {} seconds".format(str(end - start)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Run model training procedure")
    parser.add_argument("--outdir", type=str, help="Path to write output")
    parser.add_argument(
        "--n_proc", type=int, default=1, help="Path to experiment toml file"
    )
    parser.add_argument(
        "--n_iter", type=int, default=2, help="Path to experiment toml file"
    )
    parser.add_argument(
        "--inner_n_folds", type=int, default=3, help="Path to experiment toml file"
    )
    parser.add_argument(
        "--outer_n_folds", type=int, default=3, help="Path to experiment toml file"
    )
    parser.add_argument(
        "--param", type=str, default="fbeta", help="Path to experiment toml file"
    )

    args = parser.parse_args()
    run(args)
