import torch
import numpy as np
import abc
import skorch

from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR


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

from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.callbacks import EpochScoring
from skorch.callbacks import LRScheduler

from typing import Union, Tuple

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
    def __init__(self, X_train, y_train, X_valid, y_valid, scoring, refit_params, run_id: Union[str, None] = None, callback=None):
        self._model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.scoring = scoring
        self.refit_params = refit_params
        self.run_id = run_id
        self._callback = callback

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
            result.get("valid_{}".format(param_x)) for param_x in self.refit_params
        ]

        if self._callback is not None:
            self._callback.set_data(self._model)

        return tuple(score_to_optimize)


class LRObjective(TrainObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_default_params(self):
        default_params = {
            "C": 1.0,
            "random_state": SEED,
            "max_iter": 100,
            "solver": "lbfgs",
            "n_jobs": -1,
            "verbose": 1,
            "class_weight": "balanced"
        }
        return default_params

    def _get_params_for(self, model):
        return model.get_params()

    def model_init(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _clf_objective(self, trial):

        C = trial.suggest_float("C", 1e-5, 1e3, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "sag"])
        max_iter = trial.suggest_categorical("max_iter", [100])

        params = self.get_default_params()

        params.update({
            "C": C,
            "solver": solver,
            "max_iter": max_iter
        })

        clf_obj = self.model_init(**params)

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
            "verbose": 1,
            "class_weight": "balanced"
        }
        return default_params

    def _get_params_for(self, model):
        return model.get_params()

    def model_init(self, **kwargs):
        return RandomForestClassifier(**kwargs)

    def _clf_objective(self, trial):

        criterion = trial.suggest_categorical(
            "criterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int(
            "n_estimators", low=100, high=500, step=50)
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", low=1, high=10, log=True
        )
        min_samples_split = trial.suggest_int(
            "min_samples_split", low=2, high=100, log=True
        )
        max_depth = trial.suggest_categorical(
            "max_depth", [5, 8, 15, 25, 30, None])

        params = self.get_default_params()

        params.update({
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "max_depth": max_depth,
            "criterion": criterion,
        })

        clf_obj = self.model_init(**params)

        return clf_obj


class MLP(nn.Module):
    def __init__(self, input_dims=256, hidden_dims=[256, 256], dropout=0.3, output_dims=1):
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


def get_train_split_undersampled(ds, y) -> Tuple[skorch.dataset.Dataset, skorch.dataset.Dataset]:

    X = torch.vstack([x[0] for x in ds])
    y = torch.vstack([x[1] for x in ds])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                          test_size=0.1, random_state=SEED)
    raw_counts = torch.unique(y_train, return_counts=True)

    pos_mask = y_train.squeeze(dim=1) > 0
    pos_mask = pos_mask.nonzero().squeeze(dim=1)

    num_pos = len(pos_mask)

    neg_mask = y_train.squeeze(dim=1) < 1
    neg_mask = neg_mask.nonzero().squeeze(dim=1)

    permuted_idxs = torch.randperm(num_pos)

    neg_mask_sampled = neg_mask[permuted_idxs]

    pos_samples = X_train[pos_mask, :]
    pos_labels = y_train[pos_mask]

    neg_samples = X_train[neg_mask_sampled, :]
    neg_labels = y_train[neg_mask_sampled]

    X_train_sampled = torch.cat((pos_samples, neg_samples), dim=0)
    y_train_sampled = torch.cat((pos_labels, neg_labels), dim=0)

    shuffled_idx = torch.randperm(len(y_train_sampled))
    X_train_sampled = X_train_sampled[shuffled_idx, :]
    y_train_sampled = y_train_sampled[shuffled_idx]

    sampled_counts = torch.unique(y_train_sampled, return_counts=True)

    logger.info(f"Sampled down targets from {raw_counts} to {sampled_counts}.")

    train_ds = skorch.dataset.Dataset(X_train_sampled, y_train_sampled)

    valid_ds = skorch.dataset.Dataset(X_valid, y_valid)
    return train_ds, valid_ds


class MLPObjective(TrainObjective):
    def __init__(self, epochs, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs

    def get_default_params(self):
        return {}

    def _get_params_for(self, model):
        params = {}

        params.update(model.get_params_for("module"))

        return params

    def model_init(self, **kwargs):
        aucpr_scorer = get_scorers().get("AUCPR")
        aucroc_scorer = get_scorers().get("AUCROC")
        recall_scorer = get_scorers().get("recall")
        precision_scorer = get_scorers().get("precision")
        f1_scorer = get_scorers().get("f1")

        aucpr_scorer_callback = EpochScoring(
            aucpr_scorer, lower_is_better=False, on_train=False, name="valid_AUCPR")
        aucroc_scorer_callback = EpochScoring(
            aucroc_scorer, lower_is_better=False, on_train=False, name="valid_AUCROC")
        recall_scorer_callback = EpochScoring(
            recall_scorer, lower_is_better=False, on_train=False, name="valid_recall")
        precision_scorer_callback = EpochScoring(
            precision_scorer, lower_is_better=False, on_train=False, name="valid_precision")
        f1_scorer_callback = EpochScoring(
            f1_scorer, lower_is_better=False, on_train=False, name="valid_f1")

        scorers = [
            aucpr_scorer_callback,
            aucroc_scorer_callback,
            precision_scorer_callback,
            f1_scorer_callback,
            recall_scorer_callback
        ]

        lr_scheduler = LRScheduler(policy=ExponentialLR,
                                   monitor='valid_loss',
                                   gamma=0.98,
                                   verbose=True)

        early_stopping = EarlyStopping(monitor="valid_AUCPR",
                                       patience=10,
                                       threshold=0.01,
                                       threshold_mode="rel",
                                       lower_is_better=False)

        nn_callbacks = scorers + [early_stopping, lr_scheduler]

        net = NeuralNetClassifier(
            module=MLP,
            module__input_dims=self.X_train.shape[1],
            module__output_dims=1,
            max_epochs=self.epochs,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adagrad,
            optimizer__lr=0.001,
            batch_size=64,
            train_split=get_train_split_undersampled,
            callbacks=nn_callbacks,
            device=DEVICE,
            ** kwargs,
        )

        return net

    def _clf_objective(self, trial):
        dropout = trial.suggest_float("module__dropout", 0.1, 0.5, step=0.05)
        clf_obj = self.model_init(module__dropout=dropout)

        return clf_obj


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
        Xt = torch.tensor(X.copy(), dtype=torch.float32)
        yt = torch.tensor(y.copy(), dtype=torch.float32).unsqueeze(1)
        return Xt, yt

    return X, y


def create_train_objective(name, X_train, y_train, X_valid, y_valid, scoring, refit_params=["AUCPR", "AUCROC"],
                           run_id: Union[str, None] = None, **kwargs):
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

        X_valid, y_valid = transform_model_inputs(
            X_valid, y_valid, model_name=name)
        return MLPObjective(X_train=X_train,
                            y_train=y_train,
                            X_valid=X_valid,
                            y_valid=y_valid,
                            scoring=scoring,
                            refit_params=refit_params,
                            run_id=run_id,
                            epochs=300,
                            callback=kwargs.get("callback", None))
