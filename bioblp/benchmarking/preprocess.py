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


<<<<<<< HEAD
def prepare_dpi_samples(pos_df, 
                        num_negs_per_pos: Union[None, int, str] = 1,
                        entity_to_id_map: Union[None, dict] = None, 
                        relation_to_id_map: Union[None, dict] = None,
                        # map_to_kgem_ids=False,
                        filtered=True):
=======
def prepare_dpi_samples(pos_df, entity_to_id_map, relation_to_id_map,
                       num_negs_per_pos: Union[None, int, str] = 1,
                       map_to_kgem_ids=True,
                       filtered=True):
>>>>>>> add missing parameter
    """
    pos_df -> Expects dataframe with true positives in format ['src', edge', 'tgt'],
              where the entities and relations of the triple are in their string ids.
              These will be converted to KGEM integer ids at a later state
    """
<<<<<<< HEAD
    pos_neg_df = pos_df.copy()
=======
    
>>>>>>> add missing parameter
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
        
<<<<<<< HEAD
    neg_sampler = PseudoTypedNegativeSampler(mapped_triples=pos_triples.mapped_triples, 
=======
    neg_sampler = BasicNegativeSampler(mapped_triples=pos_triples.mapped_triples, 
>>>>>>> add missing parameter
                                       filtered=filtered,
                                      num_negs_per_pos=num_negs_per_pos)
    pos_batch = pos_triples.mapped_triples
    neg_triples = neg_sampler.sample(pos_batch)[0]
    
    return neg_triples


def main(args):
    bm_data_path = Path(args.bm_data_path)
    kg_triples_dir = Path(args.kg_triples_dir)
    outdir = Path(args.outdir)
    num_negs_per_pos = args.num_negs_per_pos
    bm_dataset_name = bm_data_path.name.split('.tsv')[0]

    training_triples = TriplesFactory.from_path_binary(kg_triples_dir)
    entity_to_id_map = training_triples.entity_to_id
    relation_to_id_map = training_triples.relation_to_id

    # load the benchmark data
    bm_df = pd.read_csv(bm_data_path, sep='\t', names=[COL_SOURCE, COL_EDGE, COL_TARGET])

    # generate neg samples and prepare pos-neg pairs
    logger.info(f'Generating negative samples corresponding to benchmark triples')
    pos_neg_df = prepare_dpi_samples(bm_df, 
                                     entity_to_id_map=entity_to_id_map, 
                                     relation_to_id_map= relation_to_id_map,
                                     num_negs_per_pos=num_negs_per_pos)

    # create train-test-val splits
    ### not required, taken care of in the nested cv script
    
    # save to disk
    bm_postprocessed_path = outdir.joinpath(f"benchmarks/processed/{bm_dataset_name}_p2n-1-{num_negs_per_pos}.tsv")
    logger.info(f'Writing preprocessed data to {bm_postprocessed_path}')
    pos_neg_df.to_csv(bm_postprocessed_path, sep='\t')
    logger.info('Done!')


if __name__ == "__main__":

    parser = ArgumentParser(description="Preprocess benchmark triples (E.g. DPI data) for downstream prediction task")
    parser.add_argument("--bm_data_path", type=str,
                        help="Path to pick up benchmark data")
    parser.add_argument("--kg_triples_dir", type=str, help="Directory housing kg positive triples. Needed to generate negative samples")
    parser.add_argument("--num_negs_per_pos", type=int, help="Number of negative samples to generate per positive instance")
    parser.add_argument("--outdir", type=str, help="Path to data dir to write output")

    args = parser.parse_args()
    main(args)
