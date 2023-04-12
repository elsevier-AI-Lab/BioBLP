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
from typing import Union, Tuple, Dict, List

from bioblp.logger import get_logger
from bioblp.benchmarking.config import BenchmarkTrainConfig
from bioblp.benchmarking.hpo import LRObjective
from bioblp.benchmarking.hpo import MLPObjective
from bioblp.benchmarking.hpo import RFObjective
from bioblp.benchmarking.hpo import transform_model_inputs
from bioblp.benchmarking.hpo import create_train_objective
from bioblp.benchmarking.train_utils import load_feature_data
from bioblp.benchmarking.train_utils import validate_features_exist
from bioblp.benchmarking.train_utils import get_random_string
from bioblp.benchmarking.train_utils import unique_study_prefix
from bioblp.benchmarking.train_utils import generate_study_name
from bioblp.benchmarking.train_utils import get_auc_scorers
from bioblp.benchmarking.train_utils import get_scorers
from bioblp.benchmarking.train_utils import get_model_label


logger = get_logger(__name__)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_WANDB = True


# def get_random_string(length):
#     # choose from all lowercase letter
#     characters = string.ascii_lowercase + string.digits
#     result_str = "".join(rn.choice(characters) for i in range(length))

#     return result_str


# def unique_study_prefix():
#     unique_string = get_random_string(8)
#     return unique_string


# def generate_study_name(prefix, model, fold):
#     return f"{prefix}-{model}-{fold}"


# def aupr_score(y_true, y_pred):
#     """Use AUC function to calculate the area under the curve of precision recall curve"""
#     precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
#     return auc(recall, precision)


# def get_auc_scorers():
#     scorers = {
#         "PRCURVE": make_scorer(precision_recall_curve, needs_proba=True),
#         "ROCCURVE": make_scorer(roc_curve, needs_proba=True)
#     }
#     return scorers


# def get_scorers():
#     scorers = {
#         "AUCROC": make_scorer(roc_auc_score, needs_proba=True),
#         "f1": make_scorer(fbeta_score, beta=1, average="micro"),
#         "precision": make_scorer(precision_score),
#         "recall": make_scorer(recall_score),
#         "accuracy": make_scorer(accuracy_score),
#         "AUCPR": make_scorer(aupr_score, needs_proba=True),
#     }
#     return scorers


# def get_model_label(feature: str, model: str):
#     return f"{feature}__{model}"


class ConsoleResultTracker():
    def __init__(self, file=None):
        self.file = None

    def __call__(self, study, trial_i, **kwargs):
        logger.info(
            f"============== File tracker callback | study: {study} \n, frozentrial: {trial_i}, \n kwargs: {kwargs}")


class HistoryCallback(object):
    def __init__(self, outdir, prefix):
        self.history = None
        self.prefix = prefix
        self.outdir = Path(outdir)

    def __call__(self, study, trial, **kwargs):

        #         with open(self.outdir.joinpath(f"history-{self.prefix}-{trial.number}.json"), "r") as f:
        try:
            self.history.to_file(self.outdir.joinpath(
                f"history-{self.prefix}-{trial.number}.json").resolve())

        except Exception as e:
            logger.warning(
                f"Error while saving history for {self.prefix}-{trial.number}: {e}")

        self.history = None

    def set_data(self, model):
        self.history = model.history


def model_hpo(model_label: str,
              model_clf: str,
              model_feature: str,
              feature_dir: Path,
              fold_i: str,
              train_idx: np.array,
              test_idx: np.array,
              scoring: dict,
              n_iter: int,
              refit_params: List[str],
              outdir: Path,
              study_prefix: str,
              timestamp: str,
              wandb_tag: Union[str, None] = None,
              random_state: int = SEED):

    t_start = int(time())

    scores = defaultdict(dict)
    scores[model_label] = {}
    scores[model_label]["scores"] = defaultdict(list)
    scores[model_label]["curves"] = defaultdict(list)
    scores[model_label]["params"] = []
    scores[model_label]["fold"] = fold_i

    # load feature set
    X_feat, y_feat = load_feature_data(
        feature_dir.joinpath(f"{model_feature}.pt"))

    # generate study name based on model, fold and some random word
    study_name = generate_study_name(study_prefix, model_label, fold_i)

    study = optuna.create_study(  # uses TPE sampler by default
        directions=["maximize", "maximize"],
        study_name=study_name,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_feat[train_idx, :], y_feat[train_idx], test_size=0.1, stratify=y_feat[train_idx], random_state=random_state)

    history_callback = HistoryCallback(
        outdir=outdir, prefix=f"{timestamp}-{study_name}")

    # create objective
    objective = create_train_objective(
        name=model_label,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        scoring=scoring,
        refit_params=refit_params,
        callback=history_callback if "MLP" in model_label else None
    )

    # set calbacks
    tags = [study_name, study_prefix, model_label, model_feature, timestamp]

    if wandb_tag is not None:
        tags.append(wandb_tag)

    wandb_kwargs = {"project": "bioblp-dpi-tvt",
                    "entity": "discoverylab",
                    "tags": tags,
                    "config": {
                        "model_feature": model_feature,
                        "model_clf": model_clf,
                        "model_name": model_label,
                        "study_prefix": study_prefix,
                        "fold_idx": fold_i,
                        "timestamp": timestamp
                    }
                    }

    file_tracker_callback = ConsoleResultTracker()

    # perform optimisation
    study_callbacks = [file_tracker_callback]

    if LOG_WANDB:
        tracking_metrics = refit_params

        if "MLP" in model_label:
            tracking_metrics += ["train_loss",
                                 "valid_loss", "valid_AUCPR", "AUCROC"]

        wandb_callback = WeightsAndBiasesCallback(
            metric_name=tracking_metrics, wandb_kwargs=wandb_kwargs, as_multirun=True)

        study_callbacks.append(wandb_callback)

    if "MLP" in model_label:
        study_callbacks.append(history_callback)

    study.optimize(objective,
                   n_trials=n_iter,
                   callbacks=study_callbacks)

    study_trials_df = study.trials_dataframe()
    study_trials_df["study_name"] = study_name

    # need to finish wandb run between iterations
    if LOG_WANDB:
        wandb.finish()

    t_duration = int(time()) - t_start

    logger.info(
        f"Model search for {model_label} took : {t_duration} secs.")
    #
    # Refit with best params and score
    #

    trial_with_highest_AUCPR = max(
        study.best_trials, key=lambda t: t.values[0])

    logger.info(
        f"Trial with highest AUPRC: {trial_with_highest_AUCPR}")

    logger.info(
        f"Refitting model {model_label} with best params: {trial_with_highest_AUCPR.params}...")

    # construct model from best params
    best_params_i = trial_with_highest_AUCPR.params
    model = objective.model_init(**best_params_i)

    # get params and store
    model_params = objective._get_params_for(model)
    scores[model_label]["params"].append(model_params)

    # register settings in wandb

    if LOG_WANDB:

        wandb_kwargs["tags"].append("best")
        wandb.init(**wandb_kwargs)

        wandb_kwargs.update({"config": model_params})
        wandb.config.update(wandb_kwargs["config"])

    # torch tensor transform if MLP else return same
    X_t, y_t = transform_model_inputs(X_feat, y_feat, model_name=model_label)

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

    if LOG_WANDB:
        logger.info("Logging scores to wandb...")

        wandb.log(scores_i)

        if "MLP" in model_label:
            history = model.history

            train_loss, valid_loss, valid_AUCPR = history[:, (
                'train_loss', 'valid_loss', 'valid_AUCPR')]

            for step_idx in range(0, len(train_loss)):
                wandb.log({
                    "train_loss": train_loss[step_idx],
                    "valid_loss": valid_loss[step_idx],
                    "valid_AUCPR": valid_AUCPR[step_idx]}, step=step_idx)

        labels = ["0", "1"]
        y_true = y_t[test_idx]
        y_probas = model.predict_proba(X_t[test_idx, :])

        wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)
        wandb.sklearn.plot_roc(y_true, y_probas, labels)

        wandb.finish()

    logger.info("Calculating curves...")
    # accumulate scores,
    for k_i, v_i in scores_i.items():
        scores[model_label]["scores"][k_i].append(v_i)

    curves_i = defaultdict(list)
    # compute curves
    for param, scorer in get_auc_scorers().items():
        curves_i[f"{param}"] = scorer(
            model, X_t[test_idx, :], y_t[test_idx])

        # accumulate curve scores
    for k_i, v_i in curves_i.items():
        scores[model_label]["curves"][k_i].append(v_i)

    # store model
    logger.info("Storing model...")
    joblib.dump(model, outdir.joinpath(
        f"{timestamp}-{study_name}-{model_label}.joblib"))

    return (scores, study_trials_df)


def main(model_clf: str, model_feature: str, feature_dir: str, splits_path: str, split_idx: int, n_iter: int, outdir: str,
         model_label: Union[str, None] = None, refit_params: str = "AUCPR,AUCROC", study_prefix: Union[str, None] = None,
         timestamp: Union[str, None] = None, wandb_tag: Union[str, None] = None, random_state=SEED):

    start = time()
    logger.info("Starting model train script at {}.".format(start))

    # set paths
    outdir = Path(outdir)
    feature_dir = Path(feature_dir)
    splits_path = Path(splits_path)

    outdir.mkdir(parents=True, exist_ok=True)

    # set constants
    model_label = model_label or get_model_label(
        feature=model_feature, model=model_clf)

    study_prefix = study_prefix or unique_study_prefix()
    timestamp = timestamp or str(int(time()))

    scoring = get_scorers()

    refit_params = refit_params.split(",")

    # Load splits used for study

    splits_data = torch.load(splits_path)

    fold_splits = splits_data[split_idx]
    train_idx = fold_splits["train_idx"]
    test_idx = fold_splits["test_idx"]

    scores, trials_df = model_hpo(
        model_label=model_label,
        model_clf=model_clf,
        model_feature=model_feature,
        feature_dir=feature_dir,
        fold_i=split_idx,
        train_idx=train_idx,
        test_idx=test_idx,
        scoring=scoring,
        n_iter=n_iter,
        refit_params=refit_params,
        outdir=outdir,
        study_prefix=study_prefix,
        timestamp=timestamp,
        wandb_tag=wandb_tag,
        random_state=random_state
    )

    logger.info(f"Scores for {model_label} on {split_idx}: {scores}")

    torch.save([scores], outdir.joinpath(
        f"{study_prefix}-{model_label}-{split_idx}-scores.pt"))

    # collect and store trial information
    trials_file = outdir.joinpath(
        f"{study_prefix}-{model_label}-{split_idx}-trials.csv")
    trials_df.to_csv(
        trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

    end = time()
    logger.info("Ran script in {} seconds".format(str(end - start)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Run model training procedure")
    parser.add_argument("--model_clf", type=str,
                        help="Specify classifier, from [LR, RF, MLP]")
    parser.add_argument("--model_feature", type=str,
                        help="Features")
    parser.add_argument("--feature_dir", type=str,
                        help="Path to directory holding featurised data")
    parser.add_argument("--splits_path", type=str,
                        help="Path to predefined data splits")
    parser.add_argument("--split_idx", type=int, default=0,
                        help="Index of split to use")
    parser.add_argument("--n_iter", type=int, default=10,
                        help="Number of trials in HPO")
    parser.add_argument("--refit_params", type=str, default="AUCPR,AUCROC",
                        help="Metrics to optimise")
    parser.add_argument("--outdir", type=str,
                        help="Directory to store outputs")
    parser.add_argument("--model_label", type=str, default=None,
                        help="Optional label to provide for model")
    parser.add_argument("--study_prefix", type=str, default=None,
                        help="Optional identifier for study")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Optional timestamp")
    parser.add_argument("--wandb_tag", type=str, default=None,
                        help="Optional tag to add to wand runs")
    parser.add_argument("--random_state", type=str, default=None,
                        help="Optional tag to add to wand runs")

    args = parser.parse_args()

    main(**vars(args))


# def get_data_split_callback(n_splits, random_state, shuffle=True):
#     def split_fn(X, y):

#         if n_splits == 1:
#             X_indices = torch.arange(len(X))

#             train_idx, test_idx, _, _ = train_test_split(
#                 X_indices, y, test_size=0.1, stratify=y, random_state=random_state)

#             return [(train_idx, test_idx)]

#         elif n_splits > 1:
#             cv = StratifiedKFold(
#                 n_splits=n_splits, shuffle=shuffle, random_state=random_state
#             )
#             return cv.split(X, y)
#         else:
#             raise ValueError("parameter n_splits of {n_splits}, unsupported")

#     return split_fn


# def train_job_multiprocess(models, X, y, splits_callback, feature_dir, scoring, refit_params, wandb_tag, n_iter,
#                            outdir, study_prefix, timestamp, n_jobs: int, random_state=SEED) -> Tuple[list, list]:
#     logger.info(f"Running training as multiprocessing...")

#     scores = defaultdict(list)
#     study_dfs = []

#     async_results = []
#     pool = torch.multiprocessing.Pool(processes=n_jobs)

#     for model_label, model_cfg in models.items():
#         model_feature = model_cfg.get("feature")
#         model_clf = model_cfg.get("model")

#         name = get_model_label(feature=model_feature, model=model_clf)

#         for fold_i, (train_idx, test_idx) in enumerate(splits_callback(X, y)):

#             result_i = pool.apply_async(
#                 model_hpo,
#                 (fold_i, train_idx, test_idx),
#                 dict(name=name, feature_dir=feature_dir, model_feature=model_feature, scoring=scoring, refit_params=refit_params,  wandb_tag=wandb_tag,
#                      n_iter=n_iter, outdir=outdir, study_prefix=study_prefix, model_clf=model_clf, timestamp=timestamp, random_state=random_state))

#             async_results.append((name, result_i))

#     pool.close()
#     pool.join()

#     logger.info(f"Getting results...")
#     for name, x in async_results:
#         result_i = x.get()
#         scores[name].append(result_i[0])
#         study_dfs.append(result_i[1])
#         del x

#     return scores, study_dfs


# def train_job(models, X, y, splits_callback, feature_dir, scoring, refit_params, wandb_tag, n_iter,
#               outdir, study_prefix, timestamp, random_state=SEED):

#     t_total = 0

#     scores = defaultdict(list)
#     study_dfs = []
#     results = []

#     logger.info(f"Running training in single process...")

#     for model_label, model_cfg in models.items():

#         model_feature = model_cfg.get("feature")
#         model_clf = model_cfg.get("model")
#         name = get_model_label(feature=model_feature, model=model_clf)

#         for fold_i, (train_idx, test_idx) in enumerate(splits_callback(X, y)):
#             t_start = int(time())

#             result_i = model_hpo(fold_i, train_idx, test_idx,
#                                         name=name, feature_dir=feature_dir, model_feature=model_feature, scoring=scoring, refit_params=refit_params,  wandb_tag=wandb_tag,
#                                         n_iter=n_iter, outdir=outdir, study_prefix=study_prefix, model_clf=model_clf, timestamp=timestamp, random_state=random_state)
#             results.append((name, result_i))

#             t_duration = int(time()) - t_start
#             t_total += t_duration
#             logger.info(
#                 f"Model search for {model_label} on fold {fold_i} took : {t_duration} sec. Total script time {t_total} secs.")

#     logger.info(f"Getting results...")
#     for name, result_i in results:
#         scores[name].append(result_i[0])
#         study_dfs.append(result_i[1])

#     return scores, study_dfs


# def run_training_jobs(models: Dict[str, dict],
#                       data_dir: str,
#                       X,
#                       y,
#                       scoring: dict,
#                       outdir: Path,
#                       n_splits: int = 5,
#                       n_iter: int = 10,
#                       shuffle: bool = False,
#                       random_state: int = SEED,
#                       n_jobs: int = -1,
#                       refit_params: list = ["AUCPR", "AUCROC"],
#                       verbose: int = 14,
#                       timestamp: str = None,
#                       wandb_tag: str = None
#                       ) -> dict:
#     """ Nested cross validation routine.
#         Inner cv loop performs hp optimization on all folds and surfaces

#     Parameters
#     ----------
#     conf : NestedCVArguments
#         list of (label)
#     X : np.array
#         predictor
#     y : np.ndarray
#         labels
#     scoring : dict
#         dict containing sklearn scorers
#     n_splits : int, optional
#         splits for cv, by default 5
#     n_iter : int, optional
#         number of trials within inner fold, by default 10
#     shuffle : bool, optional
#         shuffles data before cv, by default True
#     random_state : int, optional
#         seed for rng, by default SEED
#     n_jobs : int, optional
#         multiprocessing, by default 10
#     refit_params : list(str), optional
#         which metric to optimize for and return refit model, by default ['AUCPR', 'AUCROC']
#     verbose : int, optional
#         level of console feedback, by default 0
#     Returns
#     -------
#     dict
#         outer cv scores e.g. {name: scores}
#     """

#     if timestamp is None:
#         timestamp = str(int(time()))

#     study_prefix = unique_study_prefix()
#     feature_dir = Path(data_dir)

#     splits_callback = get_data_split_callback(
#         n_splits=n_splits, random_state=random_state, shuffle=shuffle)

#     if n_jobs <= 1:
#         #
#         # Single process, eg when on GPU
#         #
#         scores, study_dfs = train_job(models=models,
#                                       X=X,
#                                       y=y,
#                                       splits_callback=splits_callback,
#                                       feature_dir=feature_dir,
#                                       scoring=scoring,
#                                       refit_params=refit_params,
#                                       wandb_tag=wandb_tag,
#                                       n_iter=n_iter,
#                                       outdir=outdir,
#                                       study_prefix=study_prefix,
#                                       timestamp=timestamp,
#                                       random_state=random_state)
#     else:
#         #
#         # Multiprocessing
#         #
#         scores, study_dfs = train_job_multiprocess(models=models,
#                                                    X=X,
#                                                    y=y,
#                                                    splits_callback=splits_callback,
#                                                    feature_dir=feature_dir,
#                                                    scoring=scoring,
#                                                    refit_params=refit_params,
#                                                    wandb_tag=wandb_tag,
#                                                    n_iter=n_iter,
#                                                    outdir=outdir,
#                                                    study_prefix=study_prefix,
#                                                    timestamp=timestamp,
#                                                    random_state=random_state,
#                                                    n_jobs=n_jobs)

#     # collect and store trial information
#     trials_file = outdir.joinpath(f"{timestamp}-{study_prefix}.csv")
#     study_trials_df = pd.concat(study_dfs)
#     study_trials_df.to_csv(
#         trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

#     return scores


# def run(conf: str, n_proc: int = -1, tag: str = None, override_data_root=None, override_run_id=None, **kwargs):
#     """Perform train run"""

#     # reproducibility
#     # SEED is set as global
#     start = time()
#     logger.info("Starting model building script at {}.".format(start))

#     run_id = override_run_id or str(int(start))

#     conf_path = Path(conf)

#     config = BenchmarkTrainConfig.from_toml(conf_path, run_id=run_id)

#     if override_data_root is not None:
#         config.data_root = override_data_root

#     feature_dir = config.resolve_feature_dir()

#     models_out = config.resolve_outdir()
#     models_out.mkdir(parents=True, exist_ok=True)

#     exp_output = defaultdict(dict)
#     exp_output["config"] = asdict(config)

#     if validate_features_exist(feature_dir=feature_dir,
#                                models_conf=config.models):

#         ############
#         # Setup classifiers & pipelines
#         ############

#         X_bm, y_bm = load_feature_data(
#             feature_dir.joinpath("raw.pt"), dev_run=kwargs["dev_run"])

#         logger.info("Config contains models {}.".format(config.models))

#         scorer = get_scorers()

#         cv_scores = run_training_jobs(
#             models=config.models,
#             data_dir=feature_dir,
#             X=X_bm,
#             y=y_bm,
#             scoring=scorer,
#             n_iter=config.n_iter,
#             n_splits=config.outer_n_folds,
#             shuffle=config.shuffle,
#             n_jobs=n_proc,
#             refit_params=config.refit_params,
#             random_state=SEED,
#             outdir=models_out,
#             timestamp=run_id,
#             wandb_tag=tag
#         )

#         for algo, scores in cv_scores.items():
#             logger.info("Scores {}: {}".format(algo, scores))

#         exp_output["results"] = cv_scores

#         logger.info(exp_output)

#         file_out = models_out.joinpath(
#             "cv_scores_{}.npy".format(run_id))
#         logger.info("Saving to {}".format(file_out))
#         np.save(file_out, exp_output)

#     else:
#         logger.warning(f"Terminating...")

#     end = time()

#     logger.info("Ran script in {} seconds".format(str(end - start)))

# if __name__ == "__main__":
#     parser = ArgumentParser(description="Run model training procedure")
#     parser.add_argument("--conf", type=str,
#                         help="Path to experiment configuration")
#     parser.add_argument("--override_data_root", type=str,
#                         help="Path to root of data tree")
#     parser.add_argument("--override_run_id", type=str,
#                         help="Run id of experiment")
#     parser.add_argument(
#         "--n_proc", type=int, default=-1, help="Number of cores to use in process."
#     )
#     parser.add_argument("--tag", type=str,
#                         help="Optional tag to add to wand runs")
#     parser.add_argument("--dev_run", action='store_true',
#                         help="Quick dev run")
#     # parser.add_argument(
#     #     "--n_iter", type=int, default=2, help="Number of iterations in HPO in inner cv."
#     # )
#     # parser.add_argument(
#     #     "--inner_n_folds", type=int, default=3, help="Folds to use in inner CV"
#     # )
#     # parser.add_argument(
#     #     "--outer_n_folds", type=int, default=2, help="Folds to use in outer CV"
#     # )

#     args = parser.parse_args()
#     # run(args)
#     main(**vars(args))
