import torch
import os

import pandas as pd

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from time import time
from collections import defaultdict

from collections import defaultdict

from typing import Tuple, Dict

from bioblp.logger import get_logger
from bioblp.benchmarking.config import BenchmarkTrainConfig

from bioblp.benchmarking.train import model_hpo
from bioblp.benchmarking.train_utils import validate_features_exist
from bioblp.benchmarking.train_utils import get_scorers
from bioblp.benchmarking.train_utils import get_model_label
from bioblp.benchmarking.train_utils import unique_study_prefix
from bioblp.benchmarking.split import get_splits_iter

logger = get_logger(__name__)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_job_multiprocess(models, splits_fn, feature_dir, scoring, refit_params, wandb_tag, n_iter,
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

        for fold_i, train_idx, test_idx in splits_fn():

            result_i = pool.apply_async(
                model_hpo,
                (fold_i, train_idx, test_idx),
                dict(model_label=name,
                     model_clf=model_clf,
                     model_feature=model_feature,
                     feature_dir=feature_dir,
                     scoring=scoring,
                     n_iter=n_iter,
                     refit_params=refit_params,
                     outdir=outdir,
                     study_prefix=study_prefix,
                     timestamp=timestamp,
                     wandb_tag=wandb_tag,
                     random_state=random_state))

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


def train_job(models, splits_fn, feature_dir, scoring, refit_params, wandb_tag, n_iter,
              outdir, study_prefix, timestamp, random_state=SEED) -> Tuple[list, list]:

    t_total = 0

    scores = defaultdict(list)
    study_dfs = []
    results = []

    logger.info(f"Running training in single process...")

    for model_label, model_cfg in models.items():

        model_feature = model_cfg.get("feature")
        model_clf = model_cfg.get("model")
        name = get_model_label(feature=model_feature, model=model_clf)

        for fold_i, train_idx, test_idx in splits_fn():
            t_start = int(time())

            result_i = model_hpo(model_label=name,
                                 model_clf=model_clf,
                                 model_feature=model_feature,
                                 feature_dir=feature_dir,
                                 fold_i=fold_i,
                                 train_idx=train_idx,
                                 test_idx=test_idx,
                                 scoring=scoring,
                                 n_iter=n_iter,
                                 refit_params=refit_params,
                                 outdir=outdir,
                                 study_prefix=study_prefix,
                                 timestamp=timestamp,
                                 wandb_tag=wandb_tag,
                                 random_state=random_state)

            scores_i, trials_i = result_i

            saving_time = str(int(time()))
            torch.save([scores_i], outdir.joinpath(
                f"{study_prefix}-{name}-{fold_i}-{saving_time}-scores.pt"))

            # collect and store trial information
            trials_file = outdir.joinpath(
                f"{study_prefix}-{name}-{fold_i}-{saving_time}-trials.csv")
            trials_i.to_csv(
                trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

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
                      splits_path: Path,
                      scoring: dict,
                      outdir: Path,
                      n_iter: int = 10,
                      random_state: int = SEED,
                      n_jobs: int = -1,
                      refit_params: list = ["AUCPR", "AUCROC"],
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

    splits_fn = get_splits_iter(splits_path)

    if n_jobs <= 1:
        #
        # Single process, eg when on GPU
        #
        scores, study_dfs = train_job(models=models,
                                      splits_fn=splits_fn,
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
                                                   splits_fn=splits_fn,
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

    saving_time = str(int(time()))

    trials_file = outdir.joinpath(
        f"{study_prefix}-{saving_time}-trials.csv")
    study_trials_df = pd.concat(study_dfs)
    study_trials_df.to_csv(
        trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

    # save scores
    torch.save([scores], outdir.joinpath(
        f"{study_prefix}-{saving_time}-scores.pt"))

    return scores, study_trials_df


def run(conf: str, n_proc: int = -1, tag: str = None, override_data_root=None, override_run_id=None, **kwargs):
    """Perform train run"""

    # SEED is set as global
    start = time()
    logger.info("Starting model building script at {}.".format(start))

    run_id = override_run_id or str(int(start))

    conf_path = Path(conf)

    config = BenchmarkTrainConfig.from_toml(conf_path, run_id=run_id)

    if override_data_root is not None:
        config.data_root = Path(override_data_root)

    feature_dir = config.resolve_feature_dir()
    splits_file = config.resolve_splits_file()

    models_out = config.resolve_outdir()
    models_out.mkdir(parents=True, exist_ok=True)

    exp_output = defaultdict(dict)
    exp_output["config"] = asdict(config)

    if validate_features_exist(feature_dir=feature_dir,
                               models_conf=config.models):

        ############
        # Setup classifiers & pipelines
        ############

        logger.info("Config contains models {}.".format(config.models))

        scoring = get_scorers()

        scores, trials = run_training_jobs(
            models=config.models,
            data_dir=feature_dir,
            splits_path=splits_file,
            scoring=scoring,
            n_iter=config.n_iter,
            n_jobs=n_proc,
            refit_params=config.refit_params,
            random_state=SEED,
            outdir=models_out,
            timestamp=run_id,
            wandb_tag=tag
        )

        for algo, scores in scores.items():
            logger.info("Scores {}: {}".format(algo, scores))

        exp_output["results"] = scores

        logger.info(exp_output)

        saving_time = str(int(time()))

        file_out = models_out.joinpath(f"{run_id}-{saving_time}-metadata.pt")
        logger.info("Saving to {}".format(file_out))
        torch.save(exp_output, file_out)

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

    args = parser.parse_args()
    run(**vars(args))
