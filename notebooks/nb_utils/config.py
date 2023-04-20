import abc
import toml
import json

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd
from pathlib import Path
from pykeen.triples import TriplesFactory


DISEASE = 'disease'
DRUG = 'drug'
PROTEIN = 'protein'


ARTIFACT_REGISTRY_TOML_PATH = "/home/jovyan/BioBLP/notebooks/nb_utils/artifact_registry.toml"


class EntityType(str, Enum):
    disease = DISEASE
    drug = DRUG
    protein = PROTEIN


def load_toml(toml_path: str) -> dict:
    toml_path = Path(toml_path)
    config = {}
    with open(toml_path, "r") as f:
        config = toml.load(f)

    return config


@dataclass
class ModelRegistryConfig:
    #model_base_dir: str
    registered_model_list: List[str]
    registered_model_paths: Dict
    registered_model_training_triples_paths: Dict
    
    @classmethod
    def from_toml(cls, toml_path):
    
        def cfg_from_dict(cfg_toml_dict):
            model_root_dir = Path(cfg_toml_dict.get("model_registry_root_dir"))
            registered_model_list = list(cfg_toml_dict.get("kge_models").get("registered_models"))
            registered_model_paths = {v: model_root_dir.joinpath(f"{v}") for v in registered_model_list}
            registered_model_training_triples_paths = {k: v.joinpath("training_triples") for k,v in registered_model_paths.items()}
            cfg = {}
            cfg.update({"registered_model_list": registered_model_list})
            cfg.update({"registered_model_paths": registered_model_paths})
            cfg.update({"registered_model_training_triples_paths": registered_model_training_triples_paths})
            return cfg
        
        cfg_toml_dict = load_toml(toml_path)
        cfg = cfg_from_dict(cfg_toml_dict)

        return cls(**cfg)


# todo: delete
@dataclass
class GraphRegistryConfig:
    graph_root_dir = str
    biokgb_data_splits: Dict
    biokgb_entity_type_metadata_paths: Dict
    biokgb_entity_attribute_paths: Dict
        
    @classmethod
    def from_toml(cls, toml_path):
        cfg_toml_dict = load_toml(toml_path)

        biokgb_graph_cfg = cfg_toml_dict.get("biokgb_graph")
        biokgb_data_root_dir = Path(biokgb_graph_cfg.get("biokgb_data_root_dir"))
        biokgb_graph_dir = biokgb_data_root_dir.joinpath(biokgb_graph_cfg.get("graph_dir"))
        biokgb_properties_dir = biokgb_data_root_dir.joinpath(biokgb_graph_cfg.get("properties_dir"))
        biokgb_properties = biokgb_graph_cfg.get("properties")
        
        biokgb_data_splits = {k: biokgb_graph_dir.joinpath(v) for k,v in biokgb_graph_cfg.get("data_splits").items()}
        
        biokgb_entity_type_metadata_paths = {k: biokgb_properties_dir.joinpath(v) for k,v in biokgb_properties.get("entity_type_metadata_paths").items()}
        biokgb_entity_attribute_paths = {k: biokgb_properties_dir.joinpath(v) for k,v in biokgb_properties.get("entity_attribute_paths").items()}
        
        cfg = {}
        cfg.update({"biokgb_data_splits": biokgb_data_splits})
        cfg.update({"biokgb_entity_type_metadata_paths": biokgb_entity_type_metadata_paths})
        cfg.update({"biokgb_entity_attribute_paths": biokgb_entity_attribute_paths})
        return cls(**cfg)
    

    
@dataclass
class EvaluationConfig:
    model_name: str
    model_cfg: Dict
    test_set_list: List[str]
    test_ent_w_attribute_of_type: Enum
    
    def get_model_config(self, artifact_toml_path=ARTIFACT_REGISTRY_TOML_PATH):
        model_registry_cfg = ModelRegistryConfig.from_toml(artifact_toml_path)
        registered_models = model_registry_cfg.registered_model_list
        if self.model_name not in registered_models:
            raise ValueError(f"{self.model_name} not in {registered_models}")
        
        model_path = model_registry_cfg.registered_model_paths[self.model_name]
        training_triples_path = model_path.joinpath('training_triples')

        self.model_cfg.update({'model_name': self.model_name})
        self.model_cfg.update({'model_path': model_path})
        self.model_cfg.update({'training_triples_path': training_triples_path})       
        return self
    
    def update_test_set_dict(self, test_set_list):
        if not test_set_triples:
            test_set_list = default_test_set_list()    
        self.test_set_list = test_set_list
        
        
def get_default_biokg_test_sets(
    artifact_registry_toml_path=ARTIFACT_REGISTRY_TOML_PATH
):
    toml_cfg = load_toml(artifact_registry_toml_path)
    default_test_set_list = toml_cfg.get('test_set_lists')
    return default_test_set_list


def create_entity_attr_aware_test_sets(entity_type_w_attribute: str, 
                                       graph_cfg: GraphRegistryConfig,
                                       train: TriplesFactory, 
                                       test: TriplesFactory):
    entity_metadata_path = graph_cfg.biokgb_entity_type_metadata_paths.get(entity_type_w_attribute)
    entity_attribute_path = graph_cfg.biokgb_entity_attribute_paths.get(entity_type_w_attribute)
    
    # create a subset of biokg entities of type Protein 
    entities = pd.read_csv(entity_metadata_path, sep="\t", names=[entity_type_w_attribute, "rel", "node_type"])
    entity_set = set(entities[entity_type_w_attribute].values)
    print(f"# {entity_type_w_attribute} entities in larger biokg (pre-benchmark removal): {len(entity_set)}")

    # create a set of protein entities for which we have text descriptions
    entity_w_attr_df = pd.read_json(entity_attribute_path, orient="index").reset_index()#, sep="\t", header=0, names=["protein", "attr"])
    entity_w_attr_df.rename(columns={"index": entity_type_w_attribute, 0: "attr"}, inplace=True)
    entity_attr_set = set(entity_w_attr_df[entity_type_w_attribute].values)
    print(f"# {entity_type_w_attribute} entities for which we have attributes: {len(entity_attr_set)}")
    
    test_triples_incl_ent_prop, test_triples_excl_ent_prop = split_test_triples_conditioned_on_ent_property(train_triples=train, 
                                                                                                            typed_ent_set=entity_set, 
                                                                                                            typed_ent_with_prop_set=entity_attr_set,
                                                                                                            test_triples=test)

    return test_triples_incl_ent_prop, test_triples_excl_ent_prop
