from pathlib import Path
import pandas as pd
from bioblp.logger import get_logger
from pykeen.triples import TriplesFactory

# logger = get_logger(__name__)

COL_SOURCE = 'src'
COL_EDGE = 'edg'
COL_TARGET = 'tgt'
COL_PUBYEAR = 'pubyear'


def create_random_splits(triples: pd.DataFrame, train_ratio: float, valid_ratio: float, test_ratio: float):
    """Create train/valid/test based on random strategy
    """
    triples_array = triples[[COL_SOURCE, COL_EDGE, COL_TARGET]].values

    triples_factory = TriplesFactory.from_labeled_triples(triples_array)

    train, valid, test = triples_factory.split(
        [train_ratio, valid_ratio, test_ratio], random_state=2021)

    train_triples = pd.DataFrame(train.triples, columns=[
                                 COL_SOURCE, COL_EDGE, COL_TARGET])
    valid_triples = pd.DataFrame(valid.triples, columns=[
                                 COL_SOURCE, COL_EDGE, COL_TARGET])
    test_triples = pd.DataFrame(test.triples, columns=[
                                COL_SOURCE, COL_EDGE, COL_TARGET])

    return train_triples, valid_triples, test_triples


def save_splits(train_df, test_df, valid_df, dataset_name, out_dir):
    """TODO: unused so remove"""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    train_df.to_csv(out_dir.joinpath(
        f"{dataset_name}-train.tsv"), sep='\t', index=None)
    test_df.to_csv(out_dir.joinpath(
        f"{dataset_name}-test.tsv"), sep='\t', index=None)
    valid_df.to_csv(out_dir.joinpath(
        f"{dataset_name}-valid.tsv"), sep='\t', index=None)
    print(f"saved to {out_dir}")


def load_splits(dataset: str, data_path: str, dev_sample=False) -> (TriplesFactory, TriplesFactory, TriplesFactory):
    data_path = Path(data_path)

    training_path = data_path.joinpath(f"{dataset}-train.tsv")
    valid_path = data_path.joinpath(f"{dataset}-valid.tsv")
    test_path = data_path.joinpath(f"{dataset}-test.tsv")

    train_df = pd.read_csv(training_path, index_col=None, sep="\t", dtype=str)
    valid_df = pd.read_csv(valid_path, index_col=None, sep="\t", dtype=str)
    test_df = pd.read_csv(test_path, index_col=None, sep="\t", dtype=str)

    if dev_sample:
        dev_frac = 0.01
        train_df = train_df.sample(frac=dev_frac, random_state=2021)
        valid_df = valid_df.sample(frac=dev_frac, random_state=2021)
        test_df = test_df.sample(frac=dev_frac, random_state=2021)

    training = TriplesFactory.from_labeled_triples(
        train_df[[COL_SOURCE, COL_EDGE, COL_TARGET]].values)
    valid = TriplesFactory.from_labeled_triples(
        valid_df[[COL_SOURCE, COL_EDGE, COL_TARGET]
                 ].values, entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)
    test = TriplesFactory.from_labeled_triples(
        test_df[[COL_SOURCE, COL_EDGE, COL_TARGET]
                ].values, entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)

    return training, valid, test
