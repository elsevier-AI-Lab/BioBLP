import torch
import os
import string
import optuna
import numpy as np
import random as rn
import abc
import joblib

import wandb

from torch import nn
from torch import Tensor
from argparse import ArgumentParser
from dataclasses import asdict
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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.callbacks import EpochScoring
from typing import Union, Tuple, Dict

from bioblp.logger import get_logger


logger = get_logger(__name__)


SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def aupr_score(y_true, y_pred):
    """Use AUC function to calculate the area under the curve of precision recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


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


class OptunaTrainObjective(abc.ABC):

    def _clf_objective(self, trial):
        raise NotImplementedError

    def get_default_params(self):
        raise NotImplementedError

    def _get_params_for(self, model):
        raise NotImplementedError

    def __call__(self, trial):
        raise NotImplementedError


class TrainObjective(OptunaTrainObjective):
    def __init__(self, X_train, y_train, X_valid, y_valid, scoring, refit_params, run_id: Union[str, None] = None):
        self._model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.scoring = scoring
        self.refit_params = refit_params
        self.run_id = run_id

    def __call__(self, trial):
        self._model = self._clf_objective(trial)

        self._model.fit(self.X_train, self.y_train)

        result = {}

        for metric_i, scorer_i in self.scoring.items():
            score_i = scorer_i(self._model, self.X_valid, self.y_valid)
            result[f"valid_{metric_i}"] = score_i

        trial.set_user_attr('metrics', result)
        trial.set_user_attr('run_id', self.run_id)
        trial.set_user_attr('model_params', self._get_params_for(self._model))

        score_to_optimize = [
            result.get("valid_{}".format(param_x)).mean() for param_x in self.refit_params
        ]

        return tuple(score_to_optimize)


class CVObjective(OptunaTrainObjective):
    def __init__(self, X_train, y_train, cv, scoring, refit_params, run_id: Union[str, None] = None, n_jobs: int = -1):
        self.best_model = None
        self._model = None
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.scoring = scoring
        self.refit_params = refit_params
        self.run_id = run_id
        self.n_jobs = n_jobs

    def __call__(self, trial):
        clf_obj = self._clf_objective(trial)

        result = cross_validate(
            clf_obj,
            X=self.X_train,
            y=self.y_train,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
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


class LRObjective(TrainObjective):
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


class RFObjective(TrainObjective):
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


class MLPObjective(TrainObjective):
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
        aucpr_scorer = get_scorers().get("AUCPR")

        scorer_callback = EpochScoring(
            aucpr_scorer, lower_is_better=False, on_train=False, name="valid_AUCPR")
        early_stopping = EarlyStopping(monitor="valid_AUCPR",
                                       patience=10,
                                       threshold=0.001,
                                       threshold_mode="rel",
                                       lower_is_better=False)

        net = NeuralNetClassifier(
            module=MLP,
            module__input_dims=self.X_train.shape[1],
            module__output_dims=1,
            max_epochs=self.epochs,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adagrad,
            batch_size=128,
            # train_split=0.8,
            callbacks=[scorer_callback, early_stopping],
            device=DEVICE,
            ** kwargs,
        )

        return net

    def _clf_objective(self, trial):

        lr = trial.suggest_float("optimizer__lr", 1e-5, 1e-1, log=True)
        dropout = trial.suggest_float("module__dropout", 0.1, 0.5, step=0.1)

        clf_obj = self.model_init(optimizer__lr=lr, module__dropout=dropout)

        return clf_obj


class MLPCVObjective(CVObjective, MLPObjective):
    ...


class RFCVObjective(CVObjective, RFObjective):
    ...


class LRCVObjective(CVObjective, RFObjective):
    ...
