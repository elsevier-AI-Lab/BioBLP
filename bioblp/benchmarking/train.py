import torch
import os
import string
import optuna
import numpy as np
import random as rn
import pandas as pd
import abc

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
# rf_default_params = {
#     "n_estimators": 300,
#     "criterion": "gini",
#     "min_samples_split": 2,
#     "min_samples_leaf": 1,
#     "max_features": "sqrt",
#     "random_state": SEED,
#     "n_jobs": -1,
# }

# lr_default_params = {
#     "C": 1.0,
#     "random_state": SEED,
#     "max_iter": 1000,
#     "solver": "lbfgs",
#     "n_jobs": -1,
# }


# #
# # OPT SPACES FOR RR
# #

# rf_search_space = {
#     "criterion": ["gini", "entropy"],
#     "n_estimators": np.arange(100, 300, 50, dtype=int),
#     "min_samples_leaf": [1, 2, 5, 10],
#     "min_samples_split": [2, 5, 10, 15, 100],
#     "max_depth": [5, 8, 15, 25, 30, None],
#     "random_state": [SEED],
# }

# lr_search_space = {
#     "penalty": ["l2"],
#     "C": np.logspace(-4, 3, 8),
#     "random_state": [SEED],
#     "max_iter": [1000],
#     "solver": ["lbfgs"],
#     "n_jobs": [-1],
# }


def get_random_string(length):
    # choose from all lowercase letter
    characters = string.ascii_lowercase + string.digits
    result_str = "".join(rn.choice(characters) for i in range(length))

    return result_str


def generate_study_name(model, fold):
    unique_string = get_random_string(8)
    return f"{unique_string}-{model}-{fold}"


def get_scorers():
    scorer = {
        "AUCROC": make_scorer(roc_auc_score, needs_proba=True),
        "f1": make_scorer(fbeta_score, beta=1, average="micro"),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "accuracy": make_scorer(accuracy_score),
        "AUCPR": make_scorer(aupr_score, needs_proba=True),
    }
    return scorer


# def lr_model_init(**kwargs):
#     return LogisticRegression(**kwargs)


# def rf_model_init(**kwargs):
#     return RandomForestClassifier(**kwargs)


def get_model_init(label: str):
    models = {
        "LR": lr_model_init,
        "RF": rf_model_init
    }
    return models.get(label)


# def lr_callback(trial):
#     """
#     "max_iter": 1000,
# #     "solver": "lbfgs",
# #     "n_jobs": -1,
#     """
#     random_state = 42
#     n_jobs = -1
#     # solver = "lbfgs"

#     C = trial.suggest_float("C", 1e-5, 1e3, log=True)
#     solver = trial.suggest_categorical("solver", ["lbfgs"])
#     max_iter = trial.suggest_categorical("max_iter", [1000])

#     clf_obj = lr_model_init(
#         random_state=random_state, n_jobs=n_jobs, C=C, solver=solver, max_iter=max_iter
#     )
#     return clf_obj


# def rf_callback(trial):
#     """
#         rf_search_space = {
#         "criterion": ["gini", "entropy"],
#         "n_estimators": np.arange(100, 300, 50, dtype=int),
#         "min_samples_leaf": [1, 2, 5, 10],
#         "min_samples_split": [2, 5, 10, 15, 100],
#         "max_depth": [5, 8, 15, 25, 30, None],
#         "random_state": [SEED],
#     }
#     """

#     random_state = 42
#     n_jobs = -1

#     criterion = trial.suggest_categorical("rf_criterion", ["gini", "entropy"])
#     n_estimators = trial.suggest_int(
#         "rf_n_estimators", low=100, high=300, step=50)
#     min_samples_leaf = trial.suggest_int(
#         "rf_min_samples_leaf", low=1, high=10, log=True
#     )
#     min_samples_split = trial.suggest_int(
#         "rf_min_samples_split", low=2, high=100, log=True
#     )
#     max_depth = trial.suggest_categorical(
#         "rf_max_depth", [5, 8, 15, 25, 30, None])

#     clf_obj = rf_model_init(
#         n_estimators=n_estimators,
#         min_samples_leaf=min_samples_leaf,
#         min_samples_split=min_samples_split,
#         max_depth=max_depth,
#         criterion=criterion,
#         random_state=random_state,
#         n_jobs=n_jobs,
#     )
#     return clf_obj


class CVObjective(abc.ABC):
    def __init__(self, X_train, y_train, cv, result_tracker_callback, refit_params):
        self.best_model = None
        self._model = None

        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.result_tracker_callback = result_tracker_callback
        self.refit_params = refit_params

    def _clf_objective(self, trial):
        raise NotImplementedError

    def __call__(self, trial):
        clf_obj = self._clf_objective(trial)

        result = cross_validate(
            clf_obj,
            X=self.X_train,
            y=self.y_train,
            scoring=get_scorers(),
            cv=self.cv,
            n_jobs=-1,
            return_estimator=False,
            return_train_score=True,
            verbose=14,
        )
        self._model = clf_obj
        self.callback_result_tracker(trial.number, **result)

        score_to_optimize = [
            result.get("test_{}".format(param_x)).mean() for param_x in self.refit_params
        ]

        return tuple(score_to_optimize)

    def callback_store_best(self, study, trial):

        best_trial = max(
            study.best_trials, key=lambda t: t.values[0])

        if best_trial == trial:
            self.best_model = self._model

    def callback_result_tracker(self, *args, **kwargs):
        self.result_tracker_callback(*args, **kwargs)


class LRObjective(CVObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_init(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _clf_objective(self, trial):
        random_state = 42
        n_jobs = -1

        C = trial.suggest_float("C", 1e-5, 1e3, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs"])
        max_iter = trial.suggest_categorical("max_iter", [1000])

        clf_obj = self.model_init(
            random_state=random_state, n_jobs=n_jobs, C=C, solver=solver, max_iter=max_iter
        )
        return clf_obj


class RFObjective(CVObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_init(self, **kwargs):
        return RandomForestClassifier(**kwargs)

    def _clf_objective(self, trial):
        random_state = 42
        n_jobs = -1

        criterion = trial.suggest_categorical(
            "rcriterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int(
            "n_estimators", low=100, high=300, step=50)
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", low=1, high=10, log=True
        )
        min_samples_split = trial.suggest_int(
            "min_samples_split", low=2, high=100, log=True
        )
        max_depth = trial.suggest_categorical(
            "max_depth", [5, 8, 15, 25, 30, None])

        clf_obj = self.model_init(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return clf_obj


def create_cv_objective(name, X_train, y_train, cv, result_tracker_callback,
                        refit_params=["AUCPR", "AUCROC", "f1"],):
    if name == "LR":
        return LRObjective(X_train=X_train, y_train=y_train, cv=cv, result_tracker_callback=result_tracker_callback,
                           refit_params=refit_params)
    elif name == "RF":
        return RFObjective(X_train=X_train, y_train=y_train, cv=cv, result_tracker_callback=result_tracker_callback,
                           refit_params=refit_params)

# def create_cv_objective(
#     X_train,
#     y_train,
#     cv,
#     clf_callback,
#     result_tracker_callback,
#     refit_params=["AUCPR", "AUCROC", "f1"],
# ):
#     def cv_objective(trial):

#         clf_obj = clf_callback(trial)

#         result = cross_validate(
#             clf_obj,
#             X=X_train,
#             y=y_train,
#             scoring=get_scorers(),
#             cv=cv,
#             n_jobs=-1,
#             return_estimator=False,
#             return_train_score=True,
#             verbose=14,
#         )

#         result_tracker_callback(trial.number, **result)

#         score_to_optimize = [
#             result.get("test_{}".format(param_x)).mean() for param_x in refit_params
#         ]

#         return tuple(score_to_optimize)

#     return cv_objective


def tracker_callback_fn(trial, **kwargs):
    logger.info(f"tracker | trial: {trial}: {kwargs}")


def BestModelCallback():
    def __init__(self):
        self._model

    def __call__(self, study, trial):
        pass


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

    inner_cv = StratifiedKFold(
        n_splits=inner_n_folds, shuffle=shuffle, random_state=random_state
    )

    outer_scores = {}

    for name, clf_callback in candidates:

        outer_cv = StratifiedKFold(
            n_splits=outer_n_folds, shuffle=shuffle, random_state=random_state
        )

        outer_scores[name] = defaultdict(dict)
        outer_scores[name]["scores"] = defaultdict(list)

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

            # objective = CVObjective(
            #     clf_callback=clf_callback,
            #     X_train=X[train_idx, :],
            #     y_train=y[train_idx],
            #     cv=inner_cv,
            #     result_tracker_callback=tracker_callback_fn,
            #     refit_params=["AUCPR", "AUCROC"],
            # )

            objective = create_cv_objective(
                name=name,
                X_train=X[train_idx, :],
                y_train=y[train_idx],
                cv=inner_cv,
                # clf_callback=clf_callback,
                result_tracker_callback=tracker_callback_fn,
                refit_params=["AUCPR", "AUCROC"],
            )

            # generate study name based on model, fold and some random word?
            study = optuna.create_study(
                directions=["maximize", "maximize"],
                study_name=generate_study_name(name, fold_i),
            )

            # study.optimize(objective, n_trials=inner_n_iter,
            #                callbacks=[objective.callback_store_best])
            # best_model = objective.best_model
            # logger.info(best_model)

            study.optimize(objective, n_trials=inner_n_iter)

            trials = study.best_trials
            logger.info(trials)

            trial_with_highest_AUCPR = max(
                study.best_trials, key=lambda t: t.values[0])
            logger.info(
                f"Trial with highest AUPRC: {trial_with_highest_AUCPR}")
            logger.info(f"\tnumber: {trial_with_highest_AUCPR.number}")
            logger.info(f"\tparams: {trial_with_highest_AUCPR.params}")
            logger.info(f"\tvalues: {trial_with_highest_AUCPR.values}")

            #
            # Refit with best params and score
            #

            logger.info(
                f"Refitting model {name} with best params: {trial_with_highest_AUCPR.params}...")

            model = objective.model_init(**trial_with_highest_AUCPR.params)

            # model = model_init_fn(**trial_with_highest_AUCPR.params)

            model.fit(X[train_idx, :], y[train_idx])

            logger.info(f"Scoring  model...")
            scores_i = defaultdict(float)
            for param, scorer in scoring.items():
                scores_i[param] = scorer(model, X[test_idx, :], y[test_idx])

            # accumulate scores,
            for k_i, v_i in scores_i.items():
                outer_scores[name]["scores"][k_i].append(v_i)

            # todo
            # store best model in callback?
            # get best model and score on fold with test
            # accumulate avg scores

    # for name, gs_est in sorted(gridcvs.items()):
    #     logger.info(f"Running CV for {name}..")
    #     nested_score = cross_validate(
    #         gs_est,
    #         X=X,
    #         y=y,
    #         scoring=scoring,
    #         cv=outer_cv,
    #         n_jobs=n_jobs,
    #         return_estimator=False,
    #         return_train_score=True,
    #         verbose=10,
    #     )

    #     score_to_optimize = nested_score.get("test_{}".format(refit_param))
    #     logger.info(
    #         f"{name}: outer {refit_param} {100*score_to_optimize.mean():.2f} +/- {100*score_to_optimize.std():.2f}"
    #     )
    #     outer_scores[name] = nested_score

    return outer_scores


def aupr_score(y_true, y_pred):
    """Use AUC function to calculate the area under the curve of precision recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
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
        "Resulting shapes X_train: {}, y_train: {}".format(
            X_train.shape, y_train.shape)
    )
    logger.info("Counts in y_train: {}".format(
        np.unique(y_train, return_counts=True)))
    ############
    # Setup classifiers & pipelines
    ############

    lr_label = "LR"
    clf_lr = None

    rf_label = "RF"
    clf_rf = None

    # record default params
    # exp_output["default_params"]: {
    #     lr_label: lr_default_params,
    #     rf_label: rf_default_params,
    # }

    ############
    # Setup grids
    ############
    # exp_output["grids"]: {lr_label: lr_search_space, rf_label: rf_search_space}

    ############
    # Compare models
    ############
    candidates = [
        (lr_label, clf_lr),
        (rf_label, clf_rf)
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
    file_out = out_dir.joinpath(
        "nested_cv_scores_{}.npy".format(run_timestamp))
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
