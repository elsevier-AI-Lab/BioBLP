from abc import ABC
from argparse import ArgumentParser
from collections.abc import Callable
from collections.abc import Iterable
import json
import numpy as np
import pandas as pd 
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.models import Model
import time
import torch
from typing import Union

from bioblp.logging import get_logger
from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.benchmarking.embeddings import LookupEmbedding, create_noise_embeddings
pd.options.mode.chained_assignment = None 


logger = get_logger(__name__)

# PARAMETER KEY NAMES
MODEL = 'model'
ENTITY_TO_ID_MAP = 'entity_to_id_map'
RELATION_TO_ID_MAP = 'relation_to_id_map'

LABEL = "label"
JOINT_ENCODING = "joint_encoding"\

# Allowed featurizer mmodes
KGEM = 'kgem'
RANDOM_NOISE = 'random_noise'
TRIMODEL = 'trimodel'


def concatenate(emb1, emb2):
    n = emb1.shape[0]
    out = torch.cat((emb1, emb2), dim=1).view(n,-1)
    return out


def average(emb1, emb2):
    n = emb1.shape[0]
    out = torch.stack((emb1, emb2)).mean(dim=0).view(n, -1)
    return out


def apply_entity_pair_embedding_transform(emb1, emb2, transform:Callable):
    # might want a transform fn in the future that considers relation emb (multiclass setup)
    return transform(emb1, emb2)
    

def define_joint_encoding_transform_fn(joint_transform_operator) -> Callable:
    if joint_transform_operator == "concatenate":
        return concatenate
    elif joint_transform_operator == "average":
        return average
    else:
        logger.warning(f'Unsupproted transform function {joint_transform_operator}, defaulting to concatenation')
        return concatenate


def generate_entity_pair_joint_encoding(entity_pair_df, 
                                    model, 
                                    transform_fn: Callable = concatenate):
    
    head_ents = entity_pair_df[COL_SOURCE].values
    head_embs = get_kge_for_entity_list(head_ents, model=model)
    
    tail_ents = entity_pair_df[COL_TARGET].values
    tail_embs = get_kge_for_entity_list(tail_ents, model=model)
    
    logger.info(f"Applying transformation function: f{transform_fn}, to retrieve joint encoding for entity pair")
    joint_encodings = apply_entity_pair_embedding_transform(head_embs, tail_embs, transform=transform_fn)
    entity_pair_df[JOINT_ENCODING] = joint_encodings.detach().numpy().tolist()
    return entity_pair_df


def get_kge_for_entity_list(entity_ids: Iterable,
                            model: Model):
    entity_ids = torch.LongTensor(entity_ids)
    entity_embs = model.entity_representations[0]._embeddings(entity_ids)
    return entity_embs
          
    
def load_model_and_entity_to_id_maps(model_dir: Union[None, str, Path]=None):
    model_dir = Path(model_dir)
    logger.info(f'Loading trained model from {model_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_dir.joinpath(f"trained_model.pkl"), map_location=device)    
    training_triples = TriplesFactory.from_path_binary(model_dir.joinpath("training_triples"))
    entity_to_id_map = training_triples.entity_to_id
    relation_to_id_map = training_triples.relation_to_id
    kge_artifacts = {MODEL: model,
                    ENTITY_TO_ID_MAP: entity_to_id_map, 
                    RELATION_TO_ID_MAP: relation_to_id_map}
    return kge_artifacts



class Featurizer(ABC):

    def __init__(self, 
                joint_transform_operator: Union[None, str] = None):
        self.joint_transform_operator = joint_transform_operator
        self.metadata = {'joint_transform_operator': self.joint_transform_operator}
    
    def _prepare_data(self, data, entity_to_id_map):  #use to map to kgem ids, then the featurize_entity_pairs but can be same for all featurisers
        logger.info(f'Mapping entities to KG identifiers')
        data[COL_SOURCE] = data[COL_SOURCE].map(lambda x: entity_to_id_map.get(x, 0))
        data[COL_TARGET] = data[COL_TARGET].map(lambda x: entity_to_id_map.get(x, 0))
        return data

    def featurize_entity_pairs(self):
        raise NotImplementedError()
            
    def get_entity_representations(self):
        raise NotImplementedError()

    def _apply_joint_transform_function(self, head_repr, tail_repr, transform):
        joint_encodings = apply_entity_pair_embedding_transform(head_repr, tail_repr, transform)
        return joint_encodings

    def save(self, X, y, outdir: Union[Path, str]):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        outdir_slug = self.joint_transform_operator
        outdir = Path(outdir).joinpath(outdir_slug)
        outdir = outdir.joinpath(timestr)
        outdir.mkdir(exist_ok=True, parents=True)
        x_filepath = outdir.joinpath('X.npy')
        y_filepath = outdir.joinpath('y.npy')

        np.save(x_filepath, X)
        np.save(y_filepath, y)

        metadata_filepath = outdir.joinpath('metadata.json')
        
        with open (metadata_filepath, 'w+') as f:
            json.dump(self.metadata, f)
        logger.info(f'Saved featurized instances and correponding labels y to {outdir}')

    
class KGEMFeaturizer(Featurizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def featurize_entity_pairs(self, 
                               data,
                               model_data_path,
                               ):
        data = data.copy()
        model_data_path = Path(model_data_path)
        self.metadata.update({'featurising_model_path': str(model_data_path)})
        kge_artifacts = load_model_and_entity_to_id_maps(model_data_path)
        model = kge_artifacts.get(MODEL)  # wasteful double loading
        entity_to_id_map = kge_artifacts.get(ENTITY_TO_ID_MAP)
        
        data = self._prepare_data(data, entity_to_id_map)
        head_entities = data[COL_SOURCE].values
        tail_entities = data[COL_TARGET].values
        head_repr = self.get_entity_representations(head_entities, model=model)
        tail_repr = self.get_entity_representations(tail_entities, model=model)
        
        logger.info(f"Applying transformation function to retrieve joint encoding for entity pair")
        joint_transform_operator = define_joint_encoding_transform_fn(self.joint_transform_operator)
        joint_encodings = self._apply_joint_transform_function(head_repr, tail_repr, transform=joint_transform_operator)

        X = joint_encodings.detach().numpy()
        y = data[LABEL].values
       
        return X, y
    
    def get_entity_representations(self, entity_ids: Iterable, model: Model):
        entity_repr = get_kge_for_entity_list(entity_ids, model)
        return entity_repr
    

class LookupEmbFeaturizer(Featurizer):
    def __init__(self, dim=128, **kwargs):
        raise NotImplementedError()

    
class RandomNoiseFeaturizer(Featurizer):
    def __init__(self, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.dim=dim
    
    def featurize_entity_pairs(self, 
                               data, 
                               lookuptable_outdir: Union[None, str, Path] = None
                               ):
        
        data = data.copy()
        self.metadata.update({'featurising_model_path':str(lookuptable_outdir)})
        lookuptable_outdir = Path(lookuptable_outdir)

        entities = set(data[COL_SOURCE].values).union(data[COL_TARGET].values)
        lookup = create_noise_embeddings(entities = entities,
                                         dim=self.dim,
                                outdir=lookuptable_outdir,
                                outname='random_emb')

        entity_to_id = lookup.entity_to_id   
        data = self._prepare_data(data, entity_to_id)
        
        head_entities = data[COL_SOURCE].values
        tail_entities = data[COL_TARGET].values
        head_repr = self.get_entity_representations(head_entities, lookup=lookup)
        tail_repr = self.get_entity_representations(tail_entities, lookup=lookup)
        
        logger.info(f"Applying transformation {self.joint_transform_operator} to retrieve joint encoding for entity pair")
        joint_transform_operator = define_joint_encoding_transform_fn(self.joint_transform_operator)
        joint_encodings = self._apply_joint_transform_function(head_repr, tail_repr, transform=joint_transform_operator)
        self.X = joint_encodings.detach().numpy()
        self.y = data[LABEL].values
        return self.X, self.y
   
    def get_entity_representations(self, entity_ids, lookup):
        entity_ids = torch.LongTensor(entity_ids)
        entity_repr = lookup.embeddings[entity_ids]
        return entity_repr
    
    
def init_featurizer(entity_encoder, joint_transform_operator=None,
                   randomnoise_emb_dim=128):
    if entity_encoder == KGEM:
        featurizer = KGEMFeaturizer(joint_transform_operator=joint_transform_operator)
    
    elif entity_encoder == RANDOM_NOISE:
        featurizer = RandomNoiseFeaturizer(joint_transform_operator=joint_transform_operator,
                                           dim=randomnoise_emb_dim)
        
    elif entity_encoder == TRIMODEL:
        raise ValueError("TriModel has not been implemented yet")
        
    else:
        raise ValueError("unrecognised, try again")
    
    return featurizer
    
    
def featurize_entity_pair_dataset(df, entity_encoder, entity_pair_transform: Callable):
    
    featurizer = init_featurizer(entity_encoder)
    
    head_entities = df[COL_SOURCE]
    tail_entities = df[COL_TARGET]
    
    head_repr = featurizer.get_entity_representation(head_entities)
    tail_repr = featurizer.get_entity_representations(tail_entities)
    
    features = apply_joint_transform_function(head_repr, tail_repr, entity_pair_transform)
    
    return features


def main(args):
    joint_transform_operator = args.joint_transform_operator
    bm_data_path = Path(args.bm_data_path)
    model_or_lookup_path = Path(args.model_or_lookup_path)
    featurizer_type = args.featurizer_type
    random_noise_emb_dim = args.random_noise_emb_dim
    out_dir = Path(args.features_out_dir)
    
    bm_df = pd.read_csv(bm_data_path, sep='\t')
    featurizer = init_featurizer(featurizer_type, 
                                 joint_transform_operator=joint_transform_operator,
                                 randomnoise_emb_dim=random_noise_emb_dim
                                )

    X, y = featurizer.featurize_entity_pairs(bm_df,
                                             model_or_lookup_path,
                                            )
    featurizer.save(X, y, out_dir)

    meta = {
        'bm_data_path': str(bm_data_path),
        'featurizer_type': featurizer_type,
        'model_or_lookup_path': str(model_or_lookup_path),
        'joint_transform_operator': joint_transform_operator,
        'random_noise_emb_dim': random_noise_emb_dim
        }

    with open (out_dir.joinpath('metadata.json'), 'w+') as f:
        json.dump(meta, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="Featurize benchmark entity-pairs (E.g. DPI data) for downstream prediction task")
    parser.add_argument("--bm_data_path", "-i", type=str,
                        help="Path to pick up benchmark entity-pairwise labeled data")
    parser.add_argument("--features_out_dir", "-o", type=str,
                        help="Path to save featurized data")
    parser.add_argument("--featurizer_type", "-t", type=str,
                       help="Mode of featurization, such as KGEM, random noise embeddings, etc")
    parser.add_argument("--model_or_lookup_path", "-m", type=str,
                       help="Path to featurizer artifact such as KGE model, or random noise embeddings")
    parser.add_argument("--joint_transform_operator", "-j", type=str,
                        default='concatenate',
                       help="type of operation to use to derive an entity pair representation from individual entity vectors ")
    parser.add_argument("--random_noise_emb_dim", type=int,
                       default = 128)

    args = parser.parse_args()
    main(args)