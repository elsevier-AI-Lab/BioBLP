from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd 
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.models import Model
from typing import Union
import torch

from bioblp.logging import get_logger
from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
pd.options.mode.chained_assignment = None 


logger = get_logger(__name__)

# PARAMETER KEY NAMES
MODEL = 'model'
ENTITY_TO_ID_MAP = 'entity_to_id_map'
RELATION_TO_ID_MAP = 'relation_to_id_map'


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
    
    
def generate_entity_pair_joint_encoding(entity_pair_df, 
                                    model, 
                                    transform_fn: Callable = concatenate):
    
    JOINT_ENCODING = "joint_encoding"
    
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
