import torch
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


from bioblp.benchmarking.train_utils import load_feature_data
from bioblp.logger import get_logger
from bioblp.benchmarking.config import BenchmarkSplitConfig

from typing import Union, Tuple, Dict, List

RANDOM_STATE = 12

logger = get_logger(__name__)


def get_splits_iter(splits_path):
    def splits_iterable():
        splits_data = torch.load(splits_path)
        n = len(splits_data)

        num = 0
        while num < n:
            fold_data = splits_data[num]
            yield (fold_data["split_idx"], fold_data["train_idx"], fold_data["test_idx"])
            num += 1

    return splits_iterable


def get_split_struct(train, test, idx) -> dict:
    return {
        "train_idx": train,
        "test_idx": test,
        "split_idx": str(idx)
    }


def load_split(splits_file: Path, split_idx: int) -> Tuple[np.array, np.array]:

    splits_data = torch.load(splits_file)

    fold_splits = splits_data[split_idx]
    train_idx = fold_splits["train_idx"]
    test_idx = fold_splits["test_idx"]
    fold_idx = fold_splits["split_idx"]

    return (fold_idx, train_idx, test_idx)


def main(data, n_folds=None, outdir=None, conf=None, override_data_root=None, override_run_id=None):

    if conf is not None:
        config = BenchmarkSplitConfig.from_toml(conf, run_id=override_run_id)
        if override_data_root is not None:
            config.data_root = override_data_root

        n_folds = config.n_splits
        data_path = Path(data)
        outdir = config.resolve_outdir()
    else:
        data_path = Path(data)
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # load raw benchmark data
    X_bm, y_bm = load_feature_data(data_path)

    # generate train-test split
    logger.info("Generating train test split.")

    X_indices = torch.arange(len(X_bm))

    train_idx, test_idx, _, _ = train_test_split(
        X_indices, y_bm, test_size=0.1, stratify=y_bm, random_state=RANDOM_STATE)

    split_data = {0: get_split_struct(train_idx, test_idx, idx=0)}
    train_test_split_file = outdir.joinpath("train-test-split.pt")
    torch.save(split_data, train_test_split_file)

    # generate cv splits
    logger.info("Generating cv splits.")

    cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
    )
    splits = [(train, test, idx)
              for idx, (train, test) in enumerate(cv.split(X_bm, y_bm))]

    cv_data = {x[2]: get_split_struct(x[0], x[1], x[2]) for x in splits}

    cv_split_file = outdir.joinpath("cv-splits.pt")
    torch.save(cv_data, cv_split_file)

    logger.info("Done.")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Preprocess benchmark triples (E.g. DPI data) for downstream prediction task")

    parser.add_argument("--conf", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--data", type=str,
                        help="Path to pick up benchmark data")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="Number of cv folds to produce")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Path to data dir to write output")
    parser.add_argument("--override_data_root", type=str,
                        help="Path to root of data tree")
    parser.add_argument("--override_run_id", type=str,
                        help="Override run_id")
    args = parser.parse_args()
    main(**vars(args))
