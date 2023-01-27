from argparse import ArgumentParser
from collections.abc import Callable
import pandas as pd 
from pathlib import Path
from pykeen.sampling import BasicNegativeSampler
from pykeen.sampling import PseudoTypedNegativeSampler
from pykeen.triples import TriplesFactory
from typing import Union
import torch

from bioblp.logging import get_logger
from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET


logger = get_logger(__name__)
COL_LABEL = 'label'


def prepare_dpi_samples(pos_df, 
                        num_negs_per_pos: Union[None, int, str] = 1,
                        entity_to_id_map: Union[None, dict] = None, 
                        relation_to_id_map: Union[None, dict] = None,
                        # map_to_kgem_ids=False,
                        filtered=True):
    """
    pos_df -> Expects dataframe with true positives in format ['src', edge', 'tgt'],
              where the entities and relations of the triple are in their string ids.
              These will be converted to KGEM integer ids at a later state
    """
    pos_neg_df = pos_df.copy()
    pos_triples = TriplesFactory.from_labeled_triples(pos_df[[COL_SOURCE, COL_EDGE, COL_TARGET]].values, 
                                                      entity_to_id=entity_to_id_map, 
                                                      relation_to_id=relation_to_id_map)
    
    # returns a tensor object
    neg_triples = generate_negative_triples(pos_triples,
                                           num_negs_per_pos=num_negs_per_pos,
                                           filtered=filtered)
    
    # convert to mapped triples
    neg_triples_ = pos_triples.clone_and_exchange_triples(neg_triples.view(-1, 3))
    neg_df = pd.DataFrame(neg_triples_.triples, columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
    
    # add labels
    pos_neg_df[COL_LABEL] = 1
    neg_df[COL_LABEL] = 0
        
    # append neg samples to end, so as to retain original index of positive instances.
    # Handle indexing with more care if neg samples are generated in batches
    pos_neg_df = pd.concat([pos_neg_df, neg_df], axis=0, ignore_index=True)
    return pos_neg_df

      
def generate_negative_triples(pos_triples: TriplesFactory,
                             filtered=True,
                             num_negs_per_pos = 1):
        
    neg_sampler = PseudoTypedNegativeSampler(mapped_triples=pos_triples.mapped_triples, 
                                       filtered=filtered,
                                      num_negs_per_pos=num_negs_per_pos)
    pos_batch = pos_triples.mapped_triples
    neg_triples = neg_sampler.sample(pos_batch)[0]
    
    return neg_triples


def main(args):
    # load the benchmark data
    
    # generate neg samples and prepare pos-neg pairs
    
    # create train-test-val splits
    
    # save to disk
    
    pass


if __name__ == "__main__":

    parser = ArgumentParser(description="Preprocess benchmark triples (E.g. DPI data) for downstream prediction task")
    parser.add_argument("--data_dir", type=str,
                        help="Path to pick up data")
    parser.add_argument("--num_negs_per_pos", type=int, help="Number of negative samples to generate per positive instance")
    parser.add_argument("--outdir", type=str, help="Path to write output")

    args = parser.parse_args()
    main(args)
