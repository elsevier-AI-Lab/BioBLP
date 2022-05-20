import os
import logging
import os.path as osp
from collections import Counter
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from tqdm import tqdm
from pykeen.triples import TriplesFactory

from bioblp.data import COL_SOURCE
from bioblp.data import COL_EDGE
from bioblp.data import COL_TARGET
from bioblp.data import COL_PUBYEAR

DIR_PROCESSED = 'processed'

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_entity_relation_counts(triples: pd.DataFrame):
    """Count frequency of entities and relations across triples.
    Entities are not counted twice if there is a self-loop."""
    relation_counts = triples[COL_EDGE].value_counts()

    no_loops = triples[COL_SOURCE] != triples[COL_TARGET]
    tails_no_loops = triples[COL_TARGET].where(no_loops).dropna()
    entities = pd.concat([triples[COL_SOURCE], tails_no_loops])
    entity_counts = entities.value_counts()

    return entity_counts, relation_counts


def split_train_test_triples(triples: pd.DataFrame, ratio: float):
    """Split a dataset of triples into training and test sets, so that all
    entities in the test set are in the training set.
    Triples are removed in order starting from index 0. Edges are deleted so
    that the initial proportion of relation types is preserved in the training
    set."""
    entity_counts, relation_counts = get_entity_relation_counts(triples)
    new_relation_counts = np.floor(relation_counts * ratio).astype(int)

    train_triples = []
    test_triples = []
    removed_relation_counts = Counter()
    done = {r: count == 0 for r, count in new_relation_counts.items()}

    with tqdm(total=new_relation_counts.sum(), desc='Removing triples') as bar:
        for i in range(len(triples)):
            row = triples.iloc[i]
            head = row[COL_SOURCE]
            rel = row[COL_EDGE]
            tail = row[COL_TARGET]

            # Check that removing the entity does not remove it from the
            # training set a count larger than two is required if head == tail
            if entity_counts[head] > 2 and entity_counts[tail] > 2 and not done[rel]:
                entity_counts[head] -= 1
                entity_counts[tail] -= 1
                test_triples.append(row)

                removed_relation_counts[rel] += 1
                bar.update(1)
                if removed_relation_counts[rel] == new_relation_counts[rel]:
                    done[rel] = True
                    if all(done.values()):
                        break
            else:
                train_triples.append(row)

    test_triples = pd.DataFrame(test_triples, columns=triples.columns)
    train_triples = pd.DataFrame(train_triples, columns=triples.columns)
    # Add the rest of the triples that were not removed
    train_triples = pd.concat([train_triples, triples.iloc[i + 1:]])

    print('Done!')

    return train_triples, test_triples


def create_splits(triples_path: str, random: bool = False):
    """Create train/valid/test splits based on timestamps."""
    print('Reading triples...')
    triples = pd.read_csv(triples_path, sep='\t')
    initial_length = len(triples)

    triples = triples.dropna(subset=[COL_SOURCE, COL_EDGE, COL_TARGET,
                                     COL_PUBYEAR])
    triples[COL_PUBYEAR] = triples[COL_PUBYEAR].astype(int)

    # Sort whole dataframe first to ensure repeatability
    triples = triples.sort_values(by=list(triples.columns), kind='mergesort')

    if not random:
        # Sort by pubyear before deduplicating and removing triples!
        triples = triples.sort_values(by=COL_PUBYEAR, ascending=False,
                                      ignore_index=True, kind='mergesort')
    else:
        triples = triples.sample(frac=1, random_state=0)

    # In case of duplicates, keep most recent edge
    triples = triples.drop_duplicates(subset=[COL_SOURCE, COL_EDGE,
                                              COL_TARGET],
                                      keep='first')

    print(f'Read {initial_length:,} lines, got {len(triples):,} '
          'after keeping triples with dates and deduplicating.')

    train_triples, test_triples = split_train_test_triples(triples, ratio=0.1)

    num_test_triples = len(test_triples)
    split_idx = num_test_triples // 2
    valid_triples = test_triples.iloc[split_idx:]
    test_triples = test_triples.iloc[:split_idx]

    filename = osp.basename(triples_path)
    name, ext = osp.splitext(filename)
    data_path = osp.join(osp.dirname(osp.dirname(triples_path)), DIR_PROCESSED)

    if not osp.exists(data_path):
        os.mkdir(data_path)

    splits = {'train': train_triples,
              'valid': valid_triples,
              'test': test_triples}
    for s, dataframe in splits.items():
        out_path = osp.join(data_path, f'{name}-{s}{ext}')
        dataframe.to_csv(out_path, sep='\t', index=False)
        print(f'Saved {len(dataframe):,} triples at {out_path}')


def load_triples_array(path: str):
    """Given a path to a dataset file, extract only the colums containing
    (head, relation, tail) - i.e. the triples."""
    triples = pd.read_csv(path, sep='\t', dtype=str)
    triples = triples[[COL_SOURCE, COL_EDGE, COL_TARGET]].to_numpy()

    return triples


def load_triples_factories(data_path: str, dataset: str):
    """Load a pykeen.triples.TriplesFactory tuple for training, validation,
    and testing triples."""
    processed_path = osp.join(data_path, DIR_PROCESSED)

    train_triples = load_triples_array(osp.join(processed_path,
                                                f'{dataset}-train.tsv'))
    valid_triples = load_triples_array(osp.join(processed_path,
                                                f'{dataset}-valid.tsv'))
    test_triples = load_triples_array(osp.join(processed_path,
                                               f'{dataset}-test.tsv'))

    training = TriplesFactory.from_labeled_triples(train_triples)
    validation = TriplesFactory.from_labeled_triples(
        valid_triples,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id
    )
    testing = TriplesFactory.from_labeled_triples(
        test_triples,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id
    )

    return training, validation, testing


def reuse_existing_splits(triples_path, dataset_existing_splits):
    """"""

    triples = pd.read_csv(triples_path, sep='\t', dtype=str)
    initial_length = len(triples)
    logger.info(f"{initial_length} triples in input")

    triples = triples.dropna(subset=[COL_SOURCE, COL_EDGE, COL_TARGET,
                                     COL_PUBYEAR])
    cols = [COL_SOURCE, COL_EDGE, COL_TARGET]
    triples = triples[cols]

    filename = osp.basename(triples_path)
    name, ext = osp.splitext(filename)
    data_path = osp.join(osp.dirname(osp.dirname(triples_path)), DIR_PROCESSED)

    existing_train_path = osp.join(data_path, f'{dataset_existing_splits}-train{ext}')
    existing_val_path = osp.join(data_path, f'{dataset_existing_splits}-valid{ext}')
    existing_test_path = osp.join(data_path, f'{dataset_existing_splits}-test{ext}')

    existing_train = pd.read_csv(existing_train_path, sep='\t', dtype=str)[cols]
    existing_valid = pd.read_csv(existing_val_path, sep='\t', dtype=str)[cols]
    existing_test = pd.read_csv(existing_test_path, sep='\t', dtype=str)[cols]

    all_existing_triples = existing_train.append(existing_valid.append(
        existing_test)).sort_values(by=cols, kind='mergesort')

    logger.info(f"{len(all_existing_triples)} triples in existing {dataset_existing_splits}")

    all_existing_triples_records = set([tuple(x) for x in all_existing_triples.values])
    triple_records = [tuple(x) for x in triples.sort_values(by=cols, kind='mergesort').values]

    new_records = []
    with tqdm(total=len(triple_records), desc='Checking triple overlap') as bar:
        for i in range(len(triple_records)):
            row = triple_records[i]

            try:
                all_existing_triples_records.remove(row)
            except KeyError:
                new_records.append(row)

            bar.update(1)
            bar.set_description(
                f"Checking triple overlap. Remaining set: {len(all_existing_triples_records)}", refresh=True)

    # merge new triples plus existing train for new train
    new_triples = pd.DataFrame.from_records(new_records, columns=cols)
    train_triples = new_triples.append(existing_train)

    splits = {'train': train_triples,
              'valid': existing_valid,
              'test': existing_test}

    for s, dataframe in splits.items():
        out_path = osp.join(data_path, f'{name}-{s}{ext}')
        dataframe.to_csv(out_path, sep='\t', index=False)
        print(f'Saved {len(dataframe):,} triples at {out_path}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Split a file of triples into '
                                        'train/valid/test sets based on time.')
    parser.add_argument('file', type=str)
    parser.add_argument('--random', action='store_true',
                        help='Split randomly instead.')
    parser.add_argument('--existing_dataset_splits', type=str,
                        help='Name of existing splits (assumed to be in processed)')

    args = parser.parse_args()

    if args.existing_dataset_splits is not None:
        reuse_existing_splits(args.file, args.existing_dataset_splits)
    else:
        create_splits(args.file, args.random)
