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
from dataclasses import dataclass
from dataclasses import field


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

from typing import Union, Tuple, List, Dict

from bioblp.logging import get_logger
from bioblp.benchmarking.utils import load_toml


logger = get_logger(__name__)


SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class NestedCVArguments():
    data_root: str
    experiment_root: str
    outdir: str
    models: dict
    shuffle: bool
    refit_params: List[str]
    n_iter: int = field(default=2, metadata={"help": "Stuff"})
    inner_n_folds: int = field(default=3)
    outer_n_folds: int = field(default=3)


def parse_train_config(toml_path: str) -> dict:
    conf = load_toml(toml_path=toml_path)
    cfg = {}

    cfg["models"] = conf.get("models")

    cfg.update(conf.get("train"))

    cfg["data_root"] = conf.get("data_root")
    cfg["experiment_root"] = conf.get("experiment_root")

    return cfg


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


class TerminalResultTracker():
    def __init__(self, file=None):
        self.file = None

    def __call__(self, study, trial_i, **kwargs):
        logger.info(
            f"============== File tracker callback | study: {study} \n, frozentrial: {trial_i}, \n kwargs: {kwargs}")


class CVObjective(abc.ABC):
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


def create_cv_objective(name, X_train, y_train, scoring, cv, refit_params=["AUCPR", "AUCROC"],
                        run_id: Union[str, None] = None, n_jobs: int = -1):
    if "LR" in name:
        return LRObjective(X_train=X_train,
                           y_train=y_train,
                           scoring=scoring,
                           cv=cv,
                           refit_params=refit_params,
                           run_id=run_id,
                           n_jobs=n_jobs)
    elif "RF" in name:
        return RFObjective(X_train=X_train,
                           y_train=y_train,
                           scoring=scoring,
                           cv=cv,
                           refit_params=refit_params,
                           run_id=run_id,
                           n_jobs=n_jobs)
    elif "MLP" in name:
        X_train, y_train = transform_model_inputs(
            X_train, y_train, model_name=name)
        return MLPObjective(X_train=X_train,
                            y_train=y_train,
                            scoring=scoring,
                            cv=cv,
                            refit_params=refit_params,
                            epochs=200,
                            run_id=run_id,
                            n_jobs=n_jobs)


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


def run_nested_cv(models: Dict[str, dict],
                  data_dir: str,
                  X,
                  y,
                  scoring: dict,
                  outer_n_folds: int = 5,
                  inner_n_folds: int = 2,
                  inner_n_iter: int = 10,
                  shuffle: bool = False,
                  random_state: int = SEED,
                  n_jobs: int = -1,
                  refit_params: list = ["AUCPR", "AUCROC"],
                  verbose: int = 14,
                  outdir: Path = None,
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

    inner_cv = StratifiedKFold(
        n_splits=inner_n_folds, shuffle=shuffle, random_state=random_state
    )

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

        # use raw benchmark table to perform cf split but load specific feature set
        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

            # load feature set
            X_feat, y_feat = load_feature_data(
                feature_dir.joinpath(f"{model_feature}.pt"))

            # generate study name based on model, fold and some random word
            study_name = generate_study_name(study_prefix, name, fold_i)

            study = optuna.create_study(
                directions=["maximize", "maximize"],
                study_name=study_name,
            )

            # create objective
            objective = create_cv_objective(
                name=name,
                X_train=X_feat[train_idx, :],
                y_train=y_feat[train_idx],
                cv=inner_cv,
                scoring=scoring,
                refit_params=refit_params,
                run_id=study_name,
                n_jobs=n_jobs
            )

            # set calbacks
            tags = [study_name, study_prefix, name, model_feature]
            if wandb_tag is not None:
                tags.append(wandb_tag)

            wandb_kwargs = {"project": "bioblp-dpi",
                            "entity": "discoverylab",
                            "tags": tags,
                            "config": {
                                "model_feature": model_feature,
                                "model_clf": model_clf,
                                "model_name": name,
                                "study_prefix": study_prefix,
                                "fold_idx": fold_i
                            }
                            }

            wandb_callback = WeightsAndBiasesCallback(
                metric_name=refit_params, wandb_kwargs=wandb_kwargs, as_multirun=True)

            file_tracker_callback = TerminalResultTracker(f)

            # perform optimisation
            study.optimize(objective, n_trials=inner_n_iter,
                           callbacks=[wandb_callback, file_tracker_callback])

            if timestamp is None:
                timestamp = int(time())

            trials_file = outdir.joinpath(f"{timestamp}-{study_prefix}.csv")

            study_trials_df = study.trials_dataframe()
            study_trials_df["study_name"] = study_name

            study_trials_df.to_csv(
                trials_file, mode='a', header=not os.path.exists(trials_file), index=False)

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


def run(conf: str, n_proc: int = -1, tag: str = None, override_data_root=None, override_run_id=None):
    """Perform train run"""

    # reproducibility
    # SEED is set as global
    start = time()
    run_id = override_run_id or int(start)

    logger.info("Starting model building script at {}.".format(start))

    conf_path = Path(conf)

    conf_dict = parse_train_config(conf_path)

    if override_data_root is not None:
        conf_dict.update({"data_root": Path(override_data_root)})

    conf = NestedCVArguments(**conf_dict)

    out_root = Path(conf.experiment_root).joinpath(str(run_id))
    models_out = out_root.joinpath(conf.outdir)
    models_out.mkdir(parents=True, exist_ok=True)

    exp_output = defaultdict(dict)
    exp_output["config"] = asdict(conf)

    if validate_features_exist(feature_dir=out_root,
                               models_conf=conf.models):

        ############
        # Setup classifiers & pipelines
        ############

        X_bm, y_bm = load_feature_data(
            out_root.joinpath("raw.pt"), dev_run=args.dev_run)

        logger.info("Config contains models {}.".format(conf.models))

        scorer = get_scorers()

        nested_cv_scores = run_nested_cv(
            models=conf.models,
            data_dir=out_root,
            X=X_bm,
            y=y_bm,
            scoring=scorer,
            inner_n_folds=conf.inner_n_folds,
            inner_n_iter=conf.n_iter,
            outer_n_folds=conf.outer_n_folds,
            shuffle=conf.shuffle,
            n_jobs=n_proc,
            refit_params=conf.refit_params,
            random_state=SEED,
            outdir=models_out,
            timestamp=run_id,
            wandb_tag=tag
        )

        for algo, scores in nested_cv_scores.items():
            logger.info("Scores {}: {}".format(algo, scores))

        exp_output["results"] = nested_cv_scores

        logger.info(exp_output)

        file_out = models_out.joinpath(
            "nested_cv_scores_{}.npy".format(run_id))
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
