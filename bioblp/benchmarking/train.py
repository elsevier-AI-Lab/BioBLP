import torch
import os
import optuna
import numpy as np
import joblib
import wandb

from argparse import ArgumentParser

from pathlib import Path
from time import time
from collections import defaultdict
from optuna.integration.wandb import WeightsAndBiasesCallback

from collections import defaultdict

from sklearn.model_selection import train_test_split

from typing import Union, List

from bioblp.logger import get_logger
from bioblp.benchmarking.hpo import transform_model_inputs
from bioblp.benchmarking.hpo import create_train_objective
from bioblp.benchmarking.train_utils import load_feature_data

from bioblp.benchmarking.train_utils import unique_study_prefix
from bioblp.benchmarking.train_utils import generate_study_name
from bioblp.benchmarking.train_utils import get_auc_scorers
from bioblp.benchmarking.train_utils import get_scorers
from bioblp.benchmarking.train_utils import get_model_label
from bioblp.benchmarking.split import load_split


logger = get_logger(__name__)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_WANDB = False


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

        try:
            self.history.to_file(self.outdir.joinpath(
                f"history-{self.prefix}-{trial.number}.json").resolve())

        except Exception as e:
            logger.warning(
                f"Error while saving history for {self.prefix}-{trial.number}: {e}")

        self.history = None

    def set_data(self, model):
        self.history = model.history


def model_hpo(fold_i: str,
              train_idx: np.array,
              test_idx: np.array,
              model_label: str,
              model_clf: str,
              model_feature: str,
              feature_dir: Path,
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

    feature_dir = Path(feature_dir)

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
                    },
                    "settings": wandb.Settings(start_method="fork")
                    }

    file_tracker_callback = ConsoleResultTracker()

    # perform optimisation
    study_callbacks = [file_tracker_callback]

    if LOG_WANDB:
        tracking_metrics = refit_params

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
    model_default_params = objective.get_default_params()
    model_default_params.update(best_params_i)
    model = objective.model_init(**model_default_params)

    # get params and store
    model_params = objective._get_params_for(model)
    scores[model_label]["params"].append(model_params)

    # register settings in wandb

    if LOG_WANDB:

        wandb_kwargs["tags"].append("best")
        wandb.init(**wandb_kwargs)

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

        # wandb_kwargs.update({"config": model_params})
        wandb.config.update(model_params)

        wandb.log(scores_i)

        if "MLP" in model_label:
            history = model.history

            train_loss = history[:, 'train_loss']
            valid_loss = history[:, 'valid_loss']
            valid_AUCPR = history[:, 'valid_AUCPR']
            valid_AUCROC = history[:, 'valid_AUCROC']
            valid_precision = history[:, 'valid_precision']
            valid_f1 = history[:, "valid_f1"]
            valid_recall = history[:, "valid_recall"]

            for step_idx in range(0, len(train_loss)):
                wandb.log({
                    "train_loss": train_loss[step_idx],
                    "valid_loss": valid_loss[step_idx],
                    "valid_AUCPR": valid_AUCPR[step_idx],
                    "valid_AUCROC": valid_AUCROC[step_idx],
                    "valid_precision": valid_precision[step_idx],
                    "valid_f1": valid_f1[step_idx],
                    "valid_recall": valid_recall[step_idx]
                }, step=step_idx)

        labels = ["0", "1"]
        y_true = y_t[test_idx]
        y_probas = model.predict_proba(X_t[test_idx, :])
        y_pred = model.predict(X_t[test_idx, :])

        wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)
        wandb.sklearn.plot_roc(y_true, y_probas, labels)
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)

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

    savetime = str(int(time()))
    joblib.dump(model, outdir.joinpath(
        f"{study_name}-{timestamp}-{savetime}-model.joblib"))

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

    _, train_idx, test_idx = load_split(splits_path, split_idx=split_idx)

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
    saving_time = str(int(time()))

    torch.save([scores], outdir.joinpath(
        f"{study_prefix}-{model_label}-{split_idx}-{saving_time}-scores.pt"))

    # collect and store trial information
    trials_file = outdir.joinpath(
        f"{study_prefix}-{model_label}-{split_idx}-{saving_time}-trials.csv")
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
