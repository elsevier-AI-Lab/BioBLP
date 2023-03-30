import torch
import os
import string
import optuna
import numpy as np
import random as rn
import pandas as pd
import abc
import joblib

import wandb
import multiprocessing as mp

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
from bioblp.benchmarking.config import BenchmarkTrainConfig
from bioblp.benchmarking.hpo import LRObjective
from bioblp.benchmarking.hpo import MLPObjective
from bioblp.benchmarking.hpo import RFObjective


logger = get_logger(__name__)


SEED = 42

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


class ConsoleResultTracker():
    def __init__(self, file=None):
        self.file = None

    def __call__(self, study, trial_i, **kwargs):
        logger.info(
            f"============== File tracker callback | study: {study} \n, frozentrial: {trial_i}, \n kwargs: {kwargs}")


def create_train_objective(name, X_train, y_train, X_valid, y_valid, scoring, refit_params=["AUCPR", "AUCROC"],
                           run_id: Union[str, None] = None):
    if "LR" in name:
        return LRObjective(X_train=X_train,
                           y_train=y_train,
                           X_valid=X_valid,
                           y_valid=y_valid,
                           scoring=scoring,
                           refit_params=refit_params,
                           run_id=run_id)
    elif "RF" in name:
        return RFObjective(X_train=X_train,
                           y_train=y_train,
                           X_valid=X_valid,
                           y_valid=y_valid,
                           scoring=scoring,
                           refit_params=refit_params,
                           run_id=run_id)
    elif "MLP" in name:
        X_train, y_train = transform_model_inputs(
            X_train, y_train, model_name=name)
        return MLPObjective(X_train=X_train,
                            y_train=y_train,
                            X_valid=X_valid,
                            y_valid=y_valid,
                            scoring=scoring,
                            refit_params=refit_params,
                            run_id=run_id,
                            epochs=400)


def transform_model_inputs(X: np.array,
                           y: np.array,
                           model_name: str) -> Union[Tuple[np.array, np.array], Tuple[Tensor, Tensor]]:
    """ Transform numpy to Tensor for MLP model. Needed because MLP in pytorch implementation.

    Parameters
    ----------
    X : np.array
        Feature data
    y : np.array
        lables
    model_name : str
        label for model

    Returns
    -------
    Tuple[np.array, np.array] or  Tuple[Tensor, Tensor]
        Tensors for MLP, numpy for others.
    """
    if "MLP" in model_name:
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return Xt, yt

    return X, y


def hpo_single_model(fold_i, train_idx, test_idx, name, feature_dir, model_feature, scoring, refit_params, wandb_tag, inner_n_iter, outdir, study_prefix, model_clf, timestamp):
    outer_scores = defaultdict(dict)
    outer_scores[name] = {}

    # load feature set
    X_feat, y_feat = load_feature_data(
        feature_dir.joinpath(f"{model_feature}.pt"))

    # generate study name based on model, fold and some random word
    study_name = generate_study_name(study_prefix, name, fold_i)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        study_name=study_name,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_feat[train_idx, :], y_feat[train_idx], test_size=0.1, stratify=y_feat[train_idx], random_state=fold_i)

    # create objective
    objective = create_train_objective(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        scoring=scoring,
        refit_params=refit_params
    )

    # set calbacks
    tags = [study_name, study_prefix, name, model_feature, timestamp]

    if wandb_tag is not None:
        tags.append(wandb_tag)

    wandb_kwargs = {"project": "bioblp-dpi-s",
                    "entity": "discoverylab",
                    "tags": tags,
                    "config": {
                        "model_feature": model_feature,
                        "model_clf": model_clf,
                        "model_name": name,
                        "study_prefix": study_prefix,
                        "fold_idx": fold_i,
                        "timestamp": timestamp
                    }
                    }

    wandb_callback = WeightsAndBiasesCallback(
        metric_name=refit_params, wandb_kwargs=wandb_kwargs, as_multirun=True)

    file_tracker_callback = ConsoleResultTracker()

    # perform optimisation
    study.optimize(objective, n_trials=inner_n_iter,
                   callbacks=[wandb_callback, file_tracker_callback])

    # trials_file = outdir.joinpath(f"{timestamp}-{study_prefix}.csv")

    study_trials_df = study.trials_dataframe()
    study_trials_df["study_name"] = study_name

    # study_trials_df.to_csv(
    #     trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

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

    # construct model from best params
    best_params_i = trial_with_highest_AUCPR.params
    model = objective.model_init(**best_params_i)

    # get params and store
    model_params = objective._get_params_for(model)
    outer_scores[name]["params"].append(model_params)

    # register settings in wandb
    wandb_kwargs["tags"].append("best")
    wandb.init(**wandb_kwargs)

    wandb_kwargs.update({"config": model_params})
    wandb.config.update(wandb_kwargs["config"])

    # torch tensor transform if MLP else return same
    X_t, y_t = transform_model_inputs(X_feat, y_feat, model_name=name)

    model.fit(X_t[train_idx, :], y_t[train_idx])

    logger.info(f"Scoring  model...")
    scores_i = defaultdict(float)
    for param, scorer in scoring.items():
        # test set
        scores_i[f"test_{param}"] = scorer(
            model, X_t[test_idx, :], y_t[test_idx])

        # on train
        scores_i[f"train_{param}"] = scorer(
            model, X_t[train_idx, :], y_t[train_idx])

    wandb.log(scores_i)
    wandb.finish()

    # accumulate scores,
    for k_i, v_i in scores_i.items():
        outer_scores[name]["scores"][k_i].append(v_i)
    curves_i = defaultdict(list)

    # compute curves
    for param, scorer in get_auc_scorers().items():
        curves_i[f"{param}"] = scorer(
            model, X_t[test_idx, :], y_t[test_idx])

        # accumulate curve scores
    for k_i, v_i in curves_i.items():
        outer_scores[name]["curves"][k_i].append(v_i)

    # store model
    joblib.dump(model, outdir.joinpath(
        f"{timestamp}-{study_name}-{name}.joblib"))

    return (outer_scores, study_trials_df)


def run_cv_training(models: Dict[str, dict],
                    data_dir: str,
                    X,
                    y,
                    scoring: dict,
                    outdir: Path,
                    outer_n_folds: int = 5,
                    inner_n_folds: int = 2,
                    inner_n_iter: int = 10,
                    shuffle: bool = False,
                    random_state: int = SEED,
                    n_jobs: int = -1,
                    refit_params: list = ["AUCPR", "AUCROC"],
                    verbose: int = 14,
                    timestamp: str = None,
                    wandb_tag: str = None
                    ) -> dict:
    """ Nested cross validation routine.
        Inner cv loop performs hp optimization on all folds and surfaces

    Parameters
    ----------
    conf : NestedCVArguments
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
    refit_params : list(str), optional
        which metric to optimize for and return refit model, by default ['AUCPR', 'AUCROC']
    verbose : int, optional
        level of console feedback, by default 0
    Returns
    -------
    dict
        outer cv scores e.g. {name: scores}
    """
    if timestamp is None:
        timestamp = str(int(time()))

    outer_scores = {}

    study_prefix = unique_study_prefix()

    feature_dir = Path(data_dir)

    for model_label, model_cfg in models.items():
        model_feature = model_cfg.get("feature")
        model_clf = model_cfg.get("model")

        name = get_model_label(feature=model_feature, model=model_clf)

        outer_cv = StratifiedKFold(
            n_splits=outer_n_folds, shuffle=shuffle, random_state=random_state
        )

        outer_scores[name] = defaultdict(dict)
        outer_scores[name]["scores"] = defaultdict(list)
        outer_scores[name]["curves"] = defaultdict(list)
        outer_scores[name]["params"] = []

        pool = mp.Pool()

        # use raw benchmark table to perform cf split but load specific feature set
        scores = []
        study_dfs = []
        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

            result_i = pool.apply_async(
                hpo_single_model,
                (fold_i, train_idx, test_idx),
                name=name, feature_dir=feature_dir, model_feature=model_feature, scoring=scoring, refit_params=refit_params,  wandb_tag=wandb_tag,
                inner_n_iter=inner_n_iter, outdir=outdir, study_prefix=study_prefix, model_clf=model_clf, timestamp=timestamp)

            scores.append(result_i[0])
            study_dfs.append(result_i[1])

    trials_file = outdir.joinpath(f"{timestamp}-{study_prefix}.csv")

    study_trials_df = pd.concat(study_dfs)
    study_trials_df.to_csv(
        trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

    for scores_i in scores:
        outer_scores.update(scores_i)

    return outer_scores


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


def run(conf: str, n_proc: int = -1, tag: str = None, override_data_root=None, override_run_id=None, **kwargs):
    """Perform train run"""

    # reproducibility
    # SEED is set as global
    start = time()
    logger.info("Starting model building script at {}.".format(start))

    run_id = override_run_id or str(int(start))

    conf_path = Path(conf)

    config = BenchmarkTrainConfig.from_toml(conf_path, run_id=run_id)

    if override_data_root is not None:
        config.data_root = override_data_root

    feature_dir = config.resolve_feature_dir()

    models_out = config.resolve_outdir()
    models_out.mkdir(parents=True, exist_ok=True)

    exp_output = defaultdict(dict)
    exp_output["config"] = asdict(config)

    if validate_features_exist(feature_dir=feature_dir,
                               models_conf=config.models):

        ############
        # Setup classifiers & pipelines
        ############

        X_bm, y_bm = load_feature_data(
            feature_dir.joinpath("raw.pt"), dev_run=kwargs["dev_run"])

        logger.info("Config contains models {}.".format(config.models))

        scorer = get_scorers()

        cv_scores = run_cv_training(
            models=config.models,
            data_dir=feature_dir,
            X=X_bm,
            y=y_bm,
            scoring=scorer,
            inner_n_folds=config.inner_n_folds,
            inner_n_iter=config.n_iter,
            outer_n_folds=config.outer_n_folds,
            shuffle=config.shuffle,
            n_jobs=n_proc,
            refit_params=config.refit_params,
            random_state=SEED,
            outdir=models_out,
            timestamp=run_id,
            wandb_tag=tag
        )

        for algo, scores in cv_scores.items():
            logger.info("Scores {}: {}".format(algo, scores))

        exp_output["results"] = cv_scores

        logger.info(exp_output)

        file_out = models_out.joinpath(
            "cv_scores_{}.npy".format(run_id))
        logger.info("Saving to {}".format(file_out))
        np.save(file_out, exp_output)

    else:
        logger.warning(f"Terminating...")

    end = time()

    logger.info("Ran script in {} seconds".format(str(end - start)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Run model training procedure")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--override_data_root", type=str,
                        help="Path to root of data tree")
    parser.add_argument("--override_run_id", type=str,
                        help="Run id of experiment")
    parser.add_argument(
        "--n_proc", type=int, default=-1, help="Number of cores to use in process."
    )
    parser.add_argument("--tag", type=str,
                        help="Optional tag to add to wand runs")
    parser.add_argument("--dev_run", action='store_true',
                        help="Quick dev run")
    # parser.add_argument(
    #     "--n_iter", type=int, default=2, help="Number of iterations in HPO in inner cv."
    # )
    # parser.add_argument(
    #     "--inner_n_folds", type=int, default=3, help="Folds to use in inner CV"
    # )
    # parser.add_argument(
    #     "--outer_n_folds", type=int, default=2, help="Folds to use in outer CV"
    # )

    args = parser.parse_args()
    # run(args)
    run(**vars(args))
