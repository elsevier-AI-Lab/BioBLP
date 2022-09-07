from collections.abc import Callable
import pandas as pd 
from pathlib import Path
from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
from pykeen.triples import TriplesFactory
from typing import Union
import torch

from bioblp.logging import get_logger
from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET


logger = get_logger(__name__)
COL_LABEL = 'label'


def prepare_dpi_samples(pos_df, entity_to_id_map, relation_to_id_map,
                       num_negs_per_pos: Union[None, int, str] = 1,
                       map_to_kgem_ids=True,
                       filtered=True):
    """
    pos_df -> Expects dataframe with true positives in format ['src', edge', 'tgt'],
              where the entities and relations of the triple are in their string ids.
              These will be converted to KGEM integer ids at a later state
    """
    
    pos_triples = TriplesFactory.from_labeled_triples(pos_df[[COL_SOURCE, COL_EDGE, COL_TARGET]].values, 
                                                      entity_to_id=entity_to_id_map, 
                                                      relation_to_id=relation_to_id_map)
    if map_to_kgem_ids:
        # map positive instances to KGEM integer ids
        pos_df[COL_SOURCE] = pos_df[COL_SOURCE].map(lambda x: entity_to_id_map[x])
        pos_df[COL_TARGET] = pos_df[COL_TARGET].map(lambda x: entity_to_id_map[x])
        pos_df[COL_EDGE] = pos_df[COL_EDGE].map(lambda x: relation_to_id_map[x])

        neg_triples = generate_negative_triples(pos_triples,
                                               num_negs_per_pos=num_negs_per_pos,
                                               filtered=filtered)
        
        neg_df = pd.DataFrame(neg_triples.view(-1, 3), columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
        
        # add labels
        pos_df[COL_LABEL] = 1
        neg_df[COL_LABEL] = 0
        
        # append neg samples to end, so as to retain original index of positive instances.
        # Handle indexing with more care if neg samples are generated in batches
        pos_neg_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
        return pos_neg_df
    
    else:
        raise ValueError(f"`map_to_kgem_ids` {str(map_to_kgem_ids)} is not not supported."\
                         "Current implementation only allows output of triples mapped to respective KGEM ids")

        
def generate_negative_triples(pos_triples: TriplesFactory,
                             filtered=True,
                             num_negs_per_pos = 1):
        
    neg_sampler = BasicNegativeSampler(mapped_triples=pos_triples.mapped_triples, 
                                       filtered=filtered,
                                      num_negs_per_pos=num_negs_per_pos)
    pos_batch = pos_triples.mapped_triples
    neg_triples = neg_sampler.sample(pos_batch)[0]
    
    return neg_triples