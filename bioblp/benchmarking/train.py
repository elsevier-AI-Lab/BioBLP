import torch
import os
import string
import optuna
import numpy as np
import random as rn
import pandas as pd
import abc

import wandb

from torch import nn
from argparse import ArgumentParser
from pathlib import Path
from time import time
from collections import defaultdict
from optuna.integration.wandb import WeightsAndBiasesCallback

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
from skorch import NeuralNet
from skorch import NeuralNetClassifier

from typing import Union

from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.logging import get_logger


logger = get_logger(__name__)


SEED = 42
DATA_DIR = Path("/home/jovyan/BioBLP/data/")
DATA_SHARED = Path("/home/jovyan/workbench-shared-folder/bioblp")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class FileResultTracker():
    def __init__(self):
        self.file = None

    def __call__(self, study, trial_i, **kwargs):
        logger.info(
            f"============== File tracker callback | study: {study} \n, frozentrial: {trial_i}, \n kwargs: {kwargs}")


class CVObjective(abc.ABC):
    def __init__(self, X_train, y_train, cv, scoring, refit_params, run_id: Union[str, None] = None):
        self.best_model = None
        self._model = None

        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.scoring = scoring
        self.refit_params = refit_params
        self.run_id = run_id

    def _clf_objective(self, trial):
        raise NotImplementedError

    def get_default_params(self):
        raise NotImplementedError

    def _get_params_for(self, model):
        raise NotImplementedError

    def __call__(self, trial):
        clf_obj = self._clf_objective(trial)

        result = cross_validate(
            clf_obj,
            X=self.X_train,
            y=self.y_train,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            return_estimator=False,
            return_train_score=True,
            verbose=14,
        )

        self._model = clf_obj

        trial.set_user_attr('metrics', result)
        trial.set_user_attr('run_id', self.run_id)
        trial.set_user_attr('model_params', self._get_params_for(self._model))

        score_to_optimize = [
            result.get("test_{}".format(param_x)).mean() for param_x in self.refit_params
        ]

        return tuple(score_to_optimize)


class LRObjective(CVObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_default_params(self):
        default_params = {
            "C": 1.0,
            "random_state": SEED,
            "max_iter": 1000,
            "solver": "lbfgs",
            "n_jobs": -1,
        }
        return default_params

    def _get_params_for(self, model):
        return model.get_params()

    def model_init(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _clf_objective(self, trial):
        random_state = SEED
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

    def get_default_params(self):
        default_params = {
            "n_estimators": 300,
            "criterion": "gini",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": SEED,
            "n_jobs": -1,
        }
        return default_params

    def _get_params_for(self, model):
        return model.get_params()

    def model_init(self, **kwargs):
        return RandomForestClassifier(**kwargs)

    def _clf_objective(self, trial):
        random_state = SEED
        n_jobs = -1

        criterion = trial.suggest_categorical(
            "criterion", ["gini", "entropy"])
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


class MLP(nn.Module):
    def __init__(self, input_dims=256, hidden_dims=[256, 256], dropout=0.3, output_dims=2):
        super().__init__()
        layers = []

        # Dense 1
        layers += [
            nn.Linear(input_dims, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ]
        # Dense 2
        layers += [
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
        ]
        # Output
        layers += [
            nn.Linear(hidden_dims[1], output_dims)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, X, **kwargs):
        return self.layers(X)


class MLPObjective(CVObjective):
    def __init__(self, epochs, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs

    def get_default_params(self):
        pass

    def _get_params_for(self, model):
        params = {}

        params.update(model.get_params_for("module"))
        params.update(model.get_params_for("optimizer"))

        return params

    def model_init(self, **kwargs):
        net = NeuralNetClassifier(
            module=MLP,
            module__input_dims=self.X_train.shape[1],
            module__output_dims=1,
            max_epochs=self.epochs,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adagrad,
            batch_size=128,
            train_split=None,
            callbacks=[],
            device=DEVICE,
            **kwargs,
        )

        return net

    def _clf_objective(self, trial):

        lr = trial.suggest_float("optimizer__lr", 1e-5, 1e-1, log=True)
        dropout = trial.suggest_float("module__dropout", 0.1, 0.5, step=0.1)

        clf_obj = self.model_init(optimizer__lr=lr, module__dropout=dropout)

        return clf_obj


def create_cv_objective(name, X_train, y_train, scoring, cv,
                        refit_params=["AUCPR", "AUCROC"], run_id: Union[str, None] = None):
    if name == "LR":
        return LRObjective(X_train=X_train, y_train=y_train, scoring=scoring, cv=cv,
                           refit_params=refit_params, run_id=run_id)
    elif name == "RF":

        return RFObjective(X_train=X_train, y_train=y_train, scoring=scoring, cv=cv,
                           refit_params=refit_params, run_id=run_id)
    elif name == "MLP":

        return MLPObjective(X_train=torch.tensor(X_train, dtype=torch.float32),
                            y_train=torch.tensor(
                                y_train, dtype=torch.float32).unsqueeze(1),
                            scoring=scoring, cv=cv, refit_params=refit_params, epochs=50, run_id=run_id)


def run_nested_cv(candidates: list,
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
                  verbose: int = 14,
                  outdir: Path = None
                  ) -> dict:
    """Nested cross validation routine.
    Inner cv loop performs hp optimization on all folds and surfaces
    Parameters
    ----------
    candidates : list
        list of (label)
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

    study_prefix = unique_study_prefix()

    for name, clf_callback in candidates:

        outer_cv = StratifiedKFold(
            n_splits=outer_n_folds, shuffle=shuffle, random_state=random_state
        )

        outer_scores[name] = defaultdict(dict)
        outer_scores[name]["scores"] = defaultdict(list)

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

            # generate study name based on model, fold and some random word
            study_name = generate_study_name(study_prefix, name, fold_i)

            study = optuna.create_study(
                directions=["maximize", "maximize"],
                study_name=study_name,
            )

            # create objective
            objective = create_cv_objective(
                name=name,
                X_train=X[train_idx, :],
                y_train=y[train_idx],
                cv=inner_cv,
                scoring=scoring,
                refit_params=["AUCPR", "AUCROC"],
                run_id=study_name
            )

            # set calbacks
            wandb_kwargs = {"project": "bioblp-dpi",
                            "entity": "discoverylab",
                            "tags": [study_name, study_prefix, name]}

            wandb_callback = WeightsAndBiasesCallback(
                metric_name=["AUCPR", "AUCROC"], wandb_kwargs=wandb_kwargs, as_multirun=True)

            file_tracker_callback = FileResultTracker()

            # perform optimisation
            study.optimize(objective, n_trials=inner_n_iter,
                           callbacks=[wandb_callback, file_tracker_callback])

            trials_file = outdir.joinpath(f"{study_name}.csv")

            study.trials_dataframe().to_csv(
                trials_file, mode='a', header=not os.path.exists(trials_file))

            # need to finish wandb run between iterations
            wandb.finish()

            #
            # Refit with best params and score
            #

            trial_with_highest_AUCPR = max(
                study.best_trials, key=lambda t: t.values[0])

            logger.info(
                f"Trial with highest AUPRC: {trial_with_highest_AUCPR}")

            logger.info(
                f"Refitting model {name} with best params: {trial_with_highest_AUCPR.params}...")

            wandb_kwargs["tags"].append("best")
            wandb.init(
                **wandb_kwargs, config=trial_with_highest_AUCPR.params)

            model = objective.model_init(**trial_with_highest_AUCPR.params)

            if name == "MLP":
                # pytorch model needs tensor input
                Xt = torch.tensor(X, dtype=torch.float32)
                yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

                model.fit(Xt[train_idx, :], yt[train_idx])

                logger.info(f"Scoring  model...")
                scores_i = defaultdict(float)
                for param, scorer in scoring.items():
                    scores_i[param] = scorer(
                        model, Xt[test_idx, :], yt[test_idx])

            else:

                model.fit(X[train_idx, :], y[train_idx])

                logger.info(f"Scoring  model...")
                scores_i = defaultdict(float)

                for param, scorer in scoring.items():
                    scores_i[param] = scorer(
                        model, X[test_idx, :], y[test_idx])

            wandb.log(scores_i)
            wandb.finish()

            # accumulate scores,
            for k_i, v_i in scores_i.items():
                outer_scores[name]["scores"][k_i].append(v_i)

    return outer_scores


def run(args):
    """Perform train run"""

    # reproducibility
    SEED = 2022

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

    MLP_label = "MLP"
    clf_mlp = None

    ############
    # Compare models
    ############
    candidates = [
        # (lr_label, clf_lr),
        # (rf_label, clf_rf),
        (MLP_label, clf_mlp)

    ]

    scorer = get_scorers()

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
        outdir=out_dir
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