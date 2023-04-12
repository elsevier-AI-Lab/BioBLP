import torch
from argparse import ArgumentParser
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


from bioblp.benchmarking.train_utils import load_feature_data
from bioblp.logger import get_logger

RANDOM_STATE = 12

logger = get_logger(__name__)


def get_data_split_callback(n_splits, random_state, shuffle=True):
    def split_fn(X, y):

        if n_splits == 1:
            X_indices = torch.arange(len(X))

            train_idx, test_idx, _, _ = train_test_split(
                X_indices, y, test_size=0.1, stratify=y, random_state=random_state)

            return [(train_idx, test_idx)]

        elif n_splits > 1:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
            return cv.split(X, y)
        else:
            raise ValueError("parameter n_splits of {n_splits}, unsupported")

    return split_fn


def get_split_struct(train, test, idx) -> dict:
    return {
        "train_idx": train,
        "test_idx": test,
        "split_idx": str(idx)
    }


def main(data, n_folds, outdir):
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

    split_data = [get_split_struct(train_idx, test_idx, idx=0)]
    train_test_split_file = outdir.joinpath("train-test-split.pt")
    torch.save(split_data, train_test_split_file)

    # generate cv splits
    logger.info("Generating cv splits.")

    cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
    )
    splits = [(train, test, idx)
              for idx, (train, test) in enumerate(cv.split(X_bm, y_bm))]

    cv_data = [get_split_struct(x[0], x[1], x[2]) for x in splits]

    cv_split_file = outdir.joinpath("cv-splits.pt")
    torch.save(cv_data, cv_split_file)

    logger.info("Done.")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Preprocess benchmark triples (E.g. DPI data) for downstream prediction task")
    parser.add_argument("--data", type=str,
                        help="Path to pick up benchmark data")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cv folds to produce")
    parser.add_argument("--outdir", type=str,
                        help="Path to data dir to write output")
    args = parser.parse_args()
    main(**vars(args))
