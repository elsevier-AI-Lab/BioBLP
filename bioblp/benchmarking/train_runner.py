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
from bioblp.benchmarking.hpo import transform_model_inputs
from bioblp.benchmarking.hpo import create_train_objective
from bioblp.benchmarking.train_utils import load_feature_data
from bioblp.benchmarking.train_utils import validate_features_exist


logger = get_logger(__name__)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_WANDB = True


def train_job_multiprocess(models, X, y, splits_callback, feature_dir, scoring, refit_params, wandb_tag, n_iter,
                           outdir, study_prefix, timestamp, n_jobs: int, random_state=SEED) -> Tuple[list, list]:
    logger.info(f"Running training as multiprocessing...")

    scores = defaultdict(list)
    study_dfs = []

    async_results = []
    pool = torch.multiprocessing.Pool(processes=n_jobs)

    for model_label, model_cfg in models.items():
        model_feature = model_cfg.get("feature")
        model_clf = model_cfg.get("model")

        name = get_model_label(feature=model_feature, model=model_clf)

        for fold_i, (train_idx, test_idx) in enumerate(splits_callback(X, y)):

            result_i = pool.apply_async(
                hpo_single_model,
                (fold_i, train_idx, test_idx),
                dict(name=name, feature_dir=feature_dir, model_feature=model_feature, scoring=scoring, refit_params=refit_params,  wandb_tag=wandb_tag,
                     n_iter=n_iter, outdir=outdir, study_prefix=study_prefix, model_clf=model_clf, timestamp=timestamp, random_state=random_state))

            async_results.append((name, result_i))

    pool.close()
    pool.join()

    logger.info(f"Getting results...")
    for name, x in async_results:
        result_i = x.get()
        scores[name].append(result_i[0])
        study_dfs.append(result_i[1])
        del x

    return scores, study_dfs


def train_job(models, X, y, splits_callback, feature_dir, scoring, refit_params, wandb_tag, n_iter,
              outdir, study_prefix, timestamp, random_state=SEED):

    t_total = 0

    scores = defaultdict(list)
    study_dfs = []
    results = []

    logger.info(f"Running training in single process...")

    for model_label, model_cfg in models.items():

        model_feature = model_cfg.get("feature")
        model_clf = model_cfg.get("model")
        name = get_model_label(feature=model_feature, model=model_clf)

        for fold_i, (train_idx, test_idx) in enumerate(splits_callback(X, y)):
            t_start = int(time())

            result_i = hpo_single_model(fold_i, train_idx, test_idx,
                                        name=name, feature_dir=feature_dir, model_feature=model_feature, scoring=scoring, refit_params=refit_params,  wandb_tag=wandb_tag,
                                        n_iter=n_iter, outdir=outdir, study_prefix=study_prefix, model_clf=model_clf, timestamp=timestamp, random_state=random_state)
            results.append((name, result_i))

            t_duration = int(time()) - t_start
            t_total += t_duration
            logger.info(
                f"Model search for {model_label} on fold {fold_i} took : {t_duration} sec. Total script time {t_total} secs.")

    logger.info(f"Getting results...")
    for name, result_i in results:
        scores[name].append(result_i[0])
        study_dfs.append(result_i[1])

    return scores, study_dfs


def run_training_jobs(models: Dict[str, dict],
                      data_dir: str,
                      X,
                      y,
                      scoring: dict,
                      outdir: Path,
                      n_splits: int = 5,
                      n_iter: int = 10,
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
    n_splits : int, optional
        splits for cv, by default 5
    n_iter : int, optional
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

    study_prefix = unique_study_prefix()
    feature_dir = Path(data_dir)

    splits_callback = get_data_split_callback(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    if n_jobs <= 1:
        #
        # Single process, eg when on GPU
        #
        scores, study_dfs = train_job(models=models,
                                      X=X,
                                      y=y,
                                      splits_callback=splits_callback,
                                      feature_dir=feature_dir,
                                      scoring=scoring,
                                      refit_params=refit_params,
                                      wandb_tag=wandb_tag,
                                      n_iter=n_iter,
                                      outdir=outdir,
                                      study_prefix=study_prefix,
                                      timestamp=timestamp,
                                      random_state=random_state)
    else:
        #
        # Multiprocessing
        #
        scores, study_dfs = train_job_multiprocess(models=models,
                                                   X=X,
                                                   y=y,
                                                   splits_callback=splits_callback,
                                                   feature_dir=feature_dir,
                                                   scoring=scoring,
                                                   refit_params=refit_params,
                                                   wandb_tag=wandb_tag,
                                                   n_iter=n_iter,
                                                   outdir=outdir,
                                                   study_prefix=study_prefix,
                                                   timestamp=timestamp,
                                                   random_state=random_state,
                                                   n_jobs=n_jobs)

    # collect and store trial information
    trials_file = outdir.joinpath(f"{timestamp}-{study_prefix}.csv")
    study_trials_df = pd.concat(study_dfs)
    study_trials_df.to_csv(
        trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

    return scores


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

        cv_scores = run_training_jobs(
            models=config.models,
            data_dir=feature_dir,
            X=X_bm,
            y=y_bm,
            scoring=scorer,
            n_iter=config.n_iter,
            n_splits=config.outer_n_folds,
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

    args = parser.parse_args()
    # run(args)
    run(**vars(args))
