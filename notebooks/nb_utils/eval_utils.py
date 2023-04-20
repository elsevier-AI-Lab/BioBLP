from abc import ABC
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
from .config import GraphRegistryConfig
from .config import ModelRegistryConfig
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import seaborn as sns
import torch
from tqdm import tqdm
import wandb
import time
from abc import ABC
from dataclasses import dataclass


###### Define CONSTANTS ######
COL_SOURCE = "src"
COL_TARGET = "tgt"
COL_EDGE = "rel"
COL_NODE = "ent"
DEGREE = "degree"
IN_DEGREE = "in_degree"
OUT_DEGREE = "out_degree"


DISEASE = 'disease'
DRUG = 'drug'
PROTEIN = 'protein'


TEST = "test"
VALID = "valid"
TRAIN = "train"
DUMMY = "dummy"
TEST_RESTRICTED_DISEASE = "TEST_RESTRICTED_DISEASE"
TEST_EXCLUDING_DISEASE = "TEST_EXCLUDING_DISEASE"
VALID_RESTRICTED_DISEASE = "VALID_RESTRICTED_DISEASE"
TEST_RESTRICTED_PROTEIN = "TEST_RESTRICTED_PROTEIN"
TEST_EXCLUDING_PROTEIN = "TEST_EXCLUDING_PROTEIN"
TEST_RESTRICTED_DRUG = "TEST_RESTRICTED_DRUG"
TEST_EXCLUDING_DRUG = "TEST_EXCLUDING_DRUG"
TEST_RESTRICTED_ENT = "TEST_RESTRICTED_ENT"
TEST_EXCLUDING_ENT = "TEST_EXCLUDING_ENT"

ENT_SPECIFIC_TEST_SET_STUBS = {
    DISEASE: TEST_RESTRICTED_DISEASE,
    PROTEIN: TEST_RESTRICTED_PROTEIN,
    DRUG: TEST_RESTRICTED_DRUG    
}

##### Relations #####

COMPLEX_IN_PATHWAY = 'COMPLEX_IN_PATHWAY'
COMPLEX_TOP_LEVEL_PATHWAY = 'COMPLEX_TOP_LEVEL_PATHWAY'
DDI = 'DDI'
DISEASE_GENETIC_DISORDER = "DISEASE_GENETIC_DISORDER"
DISEASE_PATHWAY_ASSOCIATION = "DISEASE_PATHWAY_ASSOCIATION"
DPI = "DPI"
DRUG_CARRIER = "DRUG_CARRIER"
DRUG_DISEASE_ASSOCIATION = 'DRUG_DISEASE_ASSOCIATION'
DRUG_ENZYME = 'DRUG_ENZYME'
DRUG_PATHWAY_ASSOCIATION = 'DRUG_PATHWAY_ASSOCIATION'
DRUG_TARGET = 'DRUG_TARGET'
DRUG_TRANSPORTER = 'DRUG_TRANSPORTER'
MEMBER_OF_COMPLEX = 'MEMBER_OF_COMPLEX'
PPI = 'PPI'
PROTEIN_DISEASE_ASSOCIATION = 'PROTEIN_DISEASE_ASSOCIATION'
PROTEIN_PATHWAY_ASSOCIATION = 'PROTEIN_PATHWAY_ASSOCIATION'
RELATED_GENETIC_DISORDER = 'RELATED_GENETIC_DISORDER'


PROT_ASSOC_REL_NAMES = [DPI, DRUG_CARRIER, DRUG_ENZYME, 
                        DRUG_TARGET, DRUG_TRANSPORTER, MEMBER_OF_COMPLEX, 
                        PPI, PROTEIN_DISEASE_ASSOCIATION,
                        PROTEIN_PATHWAY_ASSOCIATION, RELATED_GENETIC_DISORDER]

DRUG_ASSOC_REL_NAMES = [DDI, DPI, DRUG_CARRIER, DRUG_DISEASE_ASSOCIATION, 
                        DRUG_ENZYME, DRUG_PATHWAY_ASSOCIATION, DRUG_TARGET, 
                        DRUG_TRANSPORTER]

DISEASE_ASSOC_REL_NAMES = [PROTEIN_DISEASE_ASSOCIATION,
                           DRUG_DISEASE_ASSOCIATION,
                           DISEASE_PATHWAY_ASSOCIATION,
                           DISEASE_GENETIC_DISORDER]

DEFAULT_RELATIVE_EVAL_DIR = "./metrics/"


# todo infer this programmatically, but it might require iterating through all entities
ENT_ASSOC_REL_NAMES = {
    DISEASE: {COL_SOURCE: [PROTEIN_DISEASE_ASSOCIATION,
                           DRUG_DISEASE_ASSOCIATION],
              COL_TARGET: [DISEASE_PATHWAY_ASSOCIATION,
                           DISEASE_GENETIC_DISORDER]
             },
    PROTEIN: {COL_SOURCE: [MEMBER_OF_COMPLEX, 
                           PPI,
                           PROTEIN_DISEASE_ASSOCIATION,
                           PROTEIN_PATHWAY_ASSOCIATION,
                           RELATED_GENETIC_DISORDER,
                          ],                       
              COL_TARGET: [DPI,
                           DRUG_CARRIER,
                           DRUG_ENZYME,
                           DRUG_TARGET,
                           DRUG_TRANSPORTER,
                          ],
             },
    DRUG: {COL_SOURCE: [DDI, DPI, DRUG_CARRIER, DRUG_DISEASE_ASSOCIATION, 
                        DRUG_ENZYME, DRUG_PATHWAY_ASSOCIATION, DRUG_TARGET, 
                        DRUG_TRANSPORTER],
           COL_TARGET: []
          },
}
              
    
def test_rel_list_validity(rel_list, train_triples):
    '''
    e.g.: test_rel_list_validity(ENT_ASSOC_REL_NAMES[PROTEIN][COL_TARGET], train_triples)
    '''
    for rel in rel_list:
        assert rel in train_triples.relation_to_id, f"{rel} not in training relations"


                     
# WandB stuff
WANDB_ENTITY_DISCOVERYLAB = "discoverylab"


# pykeen evaluation parameters
EVAL_NODE_BOTH = "both"
EVAL_NODE_HEAD = "head"
EVAL_NODE_TAIL = "tail"

EVAL_METRIC_REALISTIC = "realistic"
EVAL_METRIC_OPTIMISTIC = "optimistic"
EVAL_METRIC_PESSIMISTIC = "pessimistic"
EVAL_METRIC_RANKING_TYPES = {ranking_type: ranking_type for ranking_type in 
                             [EVAL_METRIC_OPTIMISTIC,EVAL_METRIC_PESSIMISTIC, EVAL_METRIC_REALISTIC]
                            }
HITS_AT_1 = "hits_at_1"
HITS_AT_3 = "hits_at_3"
HITS_AT_5 = "hits_at_5"
HITS_AT_10 = "hits_at_10"
ARITHMETIC_MEAN_RANK = "arithmetic_mean_rank"
ADJUSTED_ARITHMETIC_MEAN_RANK = "adjusted_arithmetic_mean_rank"
INVERSE_HARMONIC_MEAN_RANK = "inverse_harmonic_mean_rank"

EVAL_METRICS_CATALOG = {
    ARITHMETIC_MEAN_RANK: ARITHMETIC_MEAN_RANK,
    ADJUSTED_ARITHMETIC_MEAN_RANK: ADJUSTED_ARITHMETIC_MEAN_RANK,
    INVERSE_HARMONIC_MEAN_RANK: INVERSE_HARMONIC_MEAN_RANK,
    HITS_AT_1: HITS_AT_1,
    HITS_AT_3: HITS_AT_3,
    HITS_AT_5: HITS_AT_5,
    HITS_AT_10: HITS_AT_10
}

EVAL_METRICS_SHORTLIST = [HITS_AT_1, HITS_AT_3, 
                          HITS_AT_5, HITS_AT_10, 
                          INVERSE_HARMONIC_MEAN_RANK] 


class EvalMetricRankingType(Enum):
    optimistic = EVAL_METRIC_OPTIMISTIC
    pessimistic = EVAL_METRIC_PESSIMISTIC
    realistic = EVAL_METRIC_REALISTIC
    

def load_kge_model(model_base_path):
    '''model = load_kge_model(model_base_path = model_registry_paths[MODEL_ID])
    '''
    model_path = model_base_path.joinpath("trained_model.pkl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=torch.device(device))
    #train = TriplesFactory.from_path_binary(model_base_path.joinpath("training_triples"))
    return model
  

###### WandB utils ###### 

# requires user to login with wandb.login() call prior  to this
def setup_wandb_result_tracker(model_name:str, 
                               study_name:str, 
                               test_set_type:str, 
                               project_name: str, 
                               wandb_entity=WANDB_ENTITY_DISCOVERYLAB,
                               notes:str=""):
    '''
    sets up and returns a new wandb run
    '''

    wandb_kwargs = {"project": project_name,
                    "entity": wandb_entity,
                    "notes": notes,
                    "tags": [model_name, study_name, test_set_type]}
    run = wandb.init(**wandb_kwargs)
    return run


def run_experiment_and_log_wandb(study_name:str, 
                                 test_set_slug:str, 
                                 model_name:str,
                                 wandb_project_name:str, 
                                 eval_func,
                                 notes:None or str=None,
                                 wandb_entity_name:str=WANDB_ENTITY_DISCOVERYLAB,
                                 **eval_kwargs):
    tags = {"model_name": model_name,
            "study_name": study_name,
            "test_set_type": test_set_slug}
    metrics = eval_func(**eval_kwargs)
    run = setup_wandb_result_tracker(**tags, notes=notes, project_name=wandb_project_name, wandb_entity=wandb_entity_name)
    run.log(metrics)
    return metrics


##### evaluation logic ##### 


class LPEvaluator(ABC):
    '''
    Does not add too much. Do away with.
    This LP Evaluator currently expects KGE models with pykeen API
    for eval_metric_ranking_type options, see: 
    https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#ranking-types
    '''
    def __init__(self, 
                 eval_metric_ranking_type=None):
        
        if not eval_metric_ranking_type or eval_metric_ranking_type not in EVAL_METRIC_RANKING_TYPES.keys():
            eval_metric_ranking_type=EVAL_METRIC_REALISTIC
        
        self.eval_metric_ranking_type=eval_metric_realistic_type
        
        
    def evaluate_lp_all_rels_on_single_test_set(self,
                                                model, 
                                                eval_test_set_slug,
                                                train_triples=None,
                                                test_triples=None,
                                                valid_triples=None,
                                                eval_triple_node_endpoint=EVAL_NODE_BOTH):

        results_all_rels_dict = evaluate_lp_all_rels_on_single_test_set(model, 
                                                                        eval_test_set_slug=eval_test_set_slug,
                                                                        eval_triples=eval_triples,
                                                                        train_triples=train_triples,
                                                                        valid_triples=valid_triples,
                                                                        eval_triple_node_endpoint=EVAL_NODE_BOTH)
        # add to report/save to file?
        return results_all_rels_dict
        
        
    def evaluate_lp_relwise_on_single_test_set(self,
                                               **kwargs):
        results_relwise_dict = evaluate_lp_relwise_on_single_test_set(**kwargs)
        return results_relwise_dict
        
    
def evaluate_lp_all_rels_on_single_test_set(model, 
                                            eval_test_set_slug,
                                            eval_triples=None,
                                            train_triples=None,
                                            valid_triples=None, 
                                            eval_triple_node_endpoint=EVAL_NODE_BOTH
                                           ):
            
    evaluator = RankBasedEvaluator(filtered=True)   
    # should we filtering more triples in the disease restricted test sets?
    filtered_triples = obtain_filtered_triples(test_type=eval_test_set_slug, 
                                               train_triples=train_triples,
                                               valid_triples=valid_triples
                                              )
    eval_result = evaluator.evaluate(model, eval_triples.mapped_triples, 
                                     additional_filter_triples = filtered_triples)
    
    results_all_rels = make_results_dict_all_rel(eval_result, 
                                                 relation='All', 
                                                 relation_count=eval_triples.num_triples,
                                                 triple_endpoint=eval_triple_node_endpoint
                                                )
    return results_all_rels


def evaluate_lp_relwise_on_single_test_set(model,
                                            eval_test_set_slug,
                                            eval_triples=None,
                                            train_triples=None,
                                            valid_triples=None,    
                                           eval_triple_node_endpoint=EVAL_NODE_BOTH
                                           ):
    evaluator = RankBasedEvaluator(filtered=True)   
    result_dicts =[]
    test_triples = eval_triples  # triples_dict[eval_test_set_slug]
    additional_filter_triples = obtain_filtered_triples(test_type=eval_test_set_slug, 
                                                        train_triples=train_triples,
                                                        valid_triples=valid_triples
                                                       )
    for relation in tqdm(list(train_triples.relation_to_id)[:], desc='Evaluating over each relation'):
        triples_subset = test_triples.new_with_restriction(relations=[relation])
        if triples_subset.num_triples > 0:
            subset_result = evaluator.evaluate(model,
                                               triples_subset.mapped_triples, 
                                               additional_filter_triples=additional_filter_triples,
                                               use_tqdm=False)
            result_dicts.append({'results': subset_result, 'relation': relation, 'count': triples_subset.num_triples})
    results_df = pd.DataFrame([make_results_dict_all_rel(d['results'], d['relation'], d['count']) for d in result_dicts],
                             triple_endpoint=eval_triple_node_endpoint)
    rel_results_dict = results_df.set_index('Relation').transpose().to_dict()
    return rel_results_dict


##### format evaluation metrics #####

# todo: combine make_results_dict_all_rel and make_results_dict_rel_breakdown
def make_results_dict_all_rel(results, relation, relation_count,
                      triple_endpoint=EVAL_NODE_BOTH,
                      metric_type = EVAL_METRIC_REALISTIC):
    
    metrics_shortlist=EVAL_METRICS_SHORTLIST

    results = results.to_dict()
    results_dict = {'Relation': 'All' if not relation else relation,
                    'Count': relation_count,
                    **{metric: results[triple_endpoint][metric_type][metric] for metric in metrics_shortlist}
                   }

    return results_dict

def make_results_dict_rel_breakdown(results, relation, relation_count,
                                    triple_endpoint=EVAL_NODE_BOTH,
                                    metric_type = EVAL_METRIC_REALISTIC,
                                    metrics_shortlist=EVAL_METRICS_SHORTLIST):
    if not relation:
        relation= 'All'
    results = results.to_dict()
    results_dict = {relation: {
        'Count': relation_count,
        **{metric: results[triple_endpoint][metric_type][metric] for metric in metrics_shortlist}
    }}

    return results_dict


##### test set processing stuff #####


def create_entity_attr_aware_test_sets(entity_type_w_attribute: str, 
                                       graph_cfg: GraphRegistryConfig,
                                       train: TriplesFactory, 
                                       test: TriplesFactory):
    entity_metadata_path = graph_cfg.biokgb_entity_type_metadata_paths.get(entity_type_w_attribute)
    entity_attribute_path = graph_cfg.biokgb_entity_attribute_paths.get(entity_type_w_attribute)
    
    # create a subset of biokg entities of type {entity_type_w_attribute} 
    entities = pd.read_csv(entity_metadata_path, sep="\t", names=[entity_type_w_attribute, COL_EDGE, "node_type"])
    entity_set = set(entities[entity_type_w_attribute].values)
    print(f"# {entity_type_w_attribute} entities in larger biokg (pre-benchmark removal): {len(entity_set)}")

    # create a set of {entity_type_w_attribute} entities for which we have attr descriptions (e.g.: text for Diseases)
    # TODO: Currently a quick hack. Standardise input reading
    try:
        entity_w_attr_df = pd.read_json(entity_attribute_path, orient="index").reset_index()
        entity_w_attr_df.rename(columns={"index": entity_type_w_attribute, 0: "attr"}, inplace=True)
    except:
        entity_w_attr_df = pd.read_csv(entity_attribute_path, sep="\t", header=0, names=[entity_type_w_attribute, "attr"])
    entity_attr_set = set(entity_w_attr_df[entity_type_w_attribute].values)
    print(f"# {entity_type_w_attribute} entities for which we have attributes: {len(entity_attr_set)}")
    
    test_triples_incl_ent_prop, test_triples_excl_ent_prop = split_test_triples_conditioned_on_ent_property(train_triples=train, 
                                                                                                            typed_ent_set=entity_set, 
                                                                                                            typed_ent_with_prop_set=entity_attr_set,
                                                                                                            test_triples=test)

    return test_triples_incl_ent_prop, test_triples_excl_ent_prop


def split_train_ents_by_existance_of_properties(train_triples: TriplesFactory, 
                                                typed_ent_set: set,
                                                typed_ent_with_prop_set: set):
    assert typed_ent_with_prop_set.intersection(typed_ent_set)
    train_typed_ents = {k:v for k, v in train_triples.entity_to_id.items() if k in typed_ent_set}
    train_typed_ents_w_properties = {k:v for k, v in train_triples.entity_to_id.items() if k in typed_ent_with_prop_set}
    train_typed_ent_set = set(train_typed_ents.keys())
    train_typed_ents_w_properties_set = set(train_typed_ents_w_properties) # b==initialised by bioblp prop encoders
    #print(len(train_typed_ent_set), len(train_typed_ents_w_properties_set)) 
    return train_typed_ent_set, train_typed_ents_w_properties_set 
    
    
def split_test_triples_conditioned_on_ent_property(typed_ent_set: set,
                                                   typed_ent_with_prop_set: set,
                                                   test_triples: TriplesFactory,
                                                   train_triples: TriplesFactory,
                                                  ):
    train_typed_ent_set, train_typed_ents_w_properties_set = split_train_ents_by_existance_of_properties(train_triples, typed_ent_set, typed_ent_with_prop_set)
    
    ## create triples
    test_triples_excl_ents_w_properties = test_triples.new_with_restriction(entities=train_typed_ents_w_properties_set, invert_entity_selection=True)
    test_triples_excl_ents_w_properties_df = pd.DataFrame(test_triples_excl_ents_w_properties.triples, columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
    test_triples_df = pd.DataFrame(test_triples.triples, columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
    df = pd.merge(test_triples_df, test_triples_excl_ents_w_properties_df, how='outer', suffixes=('','_y'), indicator=True)
    test_triples_incl_ents_w_properties_df = df[df['_merge']=='left_only'][test_triples_df.columns]
    test_triples_incl_ents_w_properties = TriplesFactory.from_labeled_triples(test_triples_incl_ents_w_properties_df.values, 
                                                             relation_to_id=train_triples.relation_to_id, 
                                                             entity_to_id=train_triples.entity_to_id)
    return test_triples_incl_ents_w_properties, test_triples_excl_ents_w_properties


def obtain_filtered_triples(test_type, train_triples, valid_triples):
    if VALID in test_type:
        test_type = VALID
    else:
        test_type = TEST
    if test_type == VALID:
        filtered_triples = [train_triples.mapped_triples]
    elif test_type == TEST:
        filtered_triples = [train_triples.mapped_triples, valid_triples.mapped_triples]
    return filtered_triples


###### node degree analysis utils ###### 

def _compute_node_degrees_unidirectional(triple_df, node_endpoint_type):
    node_endpoint2degree_type = {COL_SOURCE: OUT_DEGREE,
                                     COL_TARGET: IN_DEGREE}
    degree_df = triple_df.groupby(node_endpoint_type)[COL_EDGE].agg("count").reset_index()
    degree_df.rename(columns={node_endpoint_type: COL_NODE, 
                              COL_EDGE: node_endpoint2degree_type.get(node_endpoint_type)}, 
                     inplace=True)
    return degree_df

def compute_node_degrees_in_out(triple_df):    
    # get out degrees by counting all outgoing edges (node=source)
    out_degree_df = _compute_node_degrees_unidirectional(triple_df, node_endpoint_type=COL_SOURCE)
    # get in degrees by counting all incoming edges (node=target)    
    in_degree_df = _compute_node_degrees_unidirectional(triple_df, node_endpoint_type=COL_TARGET)
    
    # combine in and out degrees for each node
    node_degree_df = out_degree_df.merge(in_degree_df, on=COL_NODE, how="outer")
    node_degree_df = node_degree_df.fillna(0)
    node_degree_df[DEGREE] = node_degree_df[IN_DEGREE]+node_degree_df[OUT_DEGREE]
    node_degree_df = node_degree_df.fillna(0)
    return node_degree_df



def get_unique_endpoint_entities_in_testset_of_given_degree(test_df, degree, node_endpoint_type=None,):
    if node_endpoint_type not in [COL_SOURCE, COL_TARGET]:
        raise ValueError(f"Invalid node end point type {node_endpoint_type}. Allowed values are '{COL_SOURCE}' and '{COL_TARGET}'")
    test_subset = test_df[test_df[f"{node_endpoint_type}_training_degree"]==degree]
    test_unique_ents = test_subset[node_endpoint_type].unique()
    return test_unique_ents


class NodeDegreeEvalAnalyser(ABC):
    '''
    should be general enough to support 3 degrees of freedom in eval:
    - model used
    - entity type with attribute (protein/molecule/disease) -> THIS IS HANDLED OUTSIDE for now
    - whether head/tail entity is being predicted (and thus keeping track of assoc rels)
    
    '''
    def __init__(self, 
                 train_triples: TriplesFactory, 
                 rels_assoc_by_node_endpoint_type_dict: dict):
        super().__init__()
        self.node_train_degree_dict = None
        self._compute_train_node_degree(train_triples=train_triples)
        self.rels_assoc_by_node_endpoint_type_dict = rels_assoc_by_node_endpoint_type_dict

    def _compute_train_node_degree(self, train_triples):
        training_df = pd.DataFrame(train_triples.triples, columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
        node_train_degree_df = compute_node_degrees_in_out(training_df)
        node_train_degree_dict = pd.Series(node_train_degree_df.degree.values,index=node_train_degree_df.ent).to_dict()
        self.node_train_degree_dict = node_train_degree_dict
    
    def prep_test_data(self, test_triples):
        # now bucket the test triples according to node degree
        test_triples_with_degree_df = pd.DataFrame(test_triples.triples, columns=[COL_SOURCE, COL_EDGE, COL_TARGET])
        test_triples_with_degree_df["src_training_degree"] = test_triples_with_degree_df[COL_SOURCE].apply(lambda node: self.node_train_degree_dict.get(node, 0))
        test_triples_with_degree_df["tgt_training_degree"] = test_triples_with_degree_df[COL_TARGET].apply(lambda node: self.node_train_degree_dict.get(node, 0))
        return test_triples_with_degree_df

    def evaluate(self,
                 entity_type_to_predict,
                 node_endpoint_to_predict, 
                 test_triples, 
                 model_id,
                 train_triples, 
                 valid_triples,
                 model_registry_cfg=None,
                 test_set_slug=TEST):
        # self.node_endpoint_to_predict = node_endpoint_to_predict
        print(f"Evaluating LP metrics on triples which have {entity_type_to_predict} in {node_endpoint_to_predict} position, using kge from model {model_id}")
        test_triples_with_degree_df = self.prep_test_data(test_triples)
        test_triples_with_degree_subset_df = self._create_test_df_subset_given_node_endpoint_type_to_predict(test_df = test_triples_with_degree_df, 
                                                                                                             entity_type_to_predict=entity_type_to_predict,
                                                                                                             node_endpoint_to_predict = node_endpoint_to_predict)
        
        results_by_node_degree = compute_metrics_over_triples_with_ent_node_endpoint(model_id=model_id, 
                                                                           test_triples_w_node_degree_df=test_triples_with_degree_subset_df, 
                                                                           test_set_slug=test_set_slug,
                                                                           entity_type_to_predict=entity_type_to_predict,
                                                                           node_endpoint_type=node_endpoint_to_predict,                                                           
                                                                           train_triples=train_triples,
                                                                           valid_triples=valid_triples,
                                                                           rels_assoc_by_node_endpoint_type_dict=self.rels_assoc_by_node_endpoint_type_dict,
                                                                      model_registry_cfg=model_registry_cfg)
        

        return results_by_node_degree


                
                
    def format_eval_results(self, eval_results, eval_triple_endpoint_list=[EVAL_NODE_HEAD, EVAL_NODE_TAIL, EVAL_NODE_BOTH], eval_metric_type=EVAL_METRIC_REALISTIC):
        results_by_node_degree_dicts = defaultdict(dict)
        for eval_triple_endpoint in eval_triple_endpoint_list:
            
            for result in eval_results:
                     results_by_node_degree_dicts[eval_triple_endpoint][result['ent_degree']] = make_results_dict_all_rel(result['results'],
                                                                                            relation=result['relation'],
                                                                                            relation_count=result['count'],
                                                                                            triple_endpoint=eval_triple_endpoint,
                                                                                            metric_type=eval_metric_type
                                                                                           )
        return results_by_node_degree_dicts

  
    def evaluate_and_format(self,
                 entity_type_to_predict,
                 node_endpoint_to_predict, 
                 test_triples, 
                 model_id,
                 train_triples, 
                 valid_triples,
                 model_registry_cfg=None,
                 test_set_slug=TEST,
                 eval_triple_endpoint_list=[EVAL_NODE_HEAD, EVAL_NODE_TAIL, EVAL_NODE_BOTH],
                 eval_metric_type=EVAL_METRIC_REALISTIC,
                 eval_out_dir=None,
                 write_results_to_file=True
):

        results_by_node_degree = self.evaluate(entity_type_to_predict=entity_type_to_predict,
                                               node_endpoint_to_predict=node_endpoint_to_predict, 
                                               test_triples=test_triples, 
                                               model_id=model_id,
                                               train_triples=train_triples, 
                                               valid_triples=valid_triples,
                                               model_registry_cfg=model_registry_cfg,
                                               test_set_slug=TEST)
        
        results_by_node_degree_dicts = self.format_eval_results(results_by_node_degree, 
                                                 eval_triple_endpoint_list=eval_triple_endpoint_list, 
                                                 eval_metric_type=eval_metric_type)
        # self.report.results_dict = results_by_node_degree_dicts
        
        if write_results_to_file:
            if not eval_out_dir:
                eval_out_dir = DEFAULT_RELATIVE_EVAL_DIR 
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            eval_out_dir = Path(eval_out_dir).joinpath(f"{model_id}/{timestr}")
            eval_out_dir.mkdir(exist_ok=True, parents=True)
            
            eval_out_file = eval_out_dir.joinpath(f"node-degree-eval.json")
            eval_metadata_file = eval_out_dir.joinpath(f"metadata.json")

            with open(eval_out_file, "w+") as f:
                print(f"Writing results to {str(eval_out_file)}")
                json.dump(results_by_node_degree_dicts, f)

            print(f"Writing results to {str(eval_metadata_file)}")
            with open(eval_metadata_file, "w+") as f:
                metadata_dict={
                    "model_id": model_id,
                    "entity_type_to_predict": entity_type_to_predict,
                    "entity_occuring_in_head_or_tail_position": node_endpoint_to_predict,
                    "eval_metric_type": eval_metric_type
                }
                json.dump(metadata_dict, f)

        return results_by_node_degree_dicts

        

    def write_to_file(self, write_results_to_file=True, eval_out_dir=None, results_by_node_degree_dicts=None,
                     entity_type_to_predict=DRUG, node_endpoint_to_predict=COL_SOURCE):
                
        if write_results_to_file:
            if not eval_out_dir:
                eval_out_dir = DEFAULT_RELATIVE_EVAL_DIR 
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            eval_out_dir = Path(eval_out_dir).joinpath(f"{model_id}/{timestr}")
            eval_out_dir.mkdir(exist_ok=True, parents=True)
            
            eval_out_file = eval_out_dir.joinpath("node-degree-eval.json")
            eval_metadata_file = eval_out_dir.joinpath("metadata.json")

            print(f"Writing results to {str(eval_out_file)}")
            with open(eval_out_file, "w+") as f:
                json.dump(results_by_node_degree_dicts, f)
            
            with open(eval_metadata_file, "w+") as f:
                metadata_dict={
                    "model_id": model_id,
                    "entity_type_to_predict": entity_type_to_predict,
                    "node_endpoint_to_predict": node_endpoint_to_predict,
                    "eval_metric_type": eval_metric_type
                }
                json.dump(metadata_dict, f)
                
     
    def _create_test_df_subset_given_node_endpoint_type_to_predict(self, 
                                                                   test_df,
                                                                   entity_type_to_predict,
                                                                   node_endpoint_to_predict):
        test_subset_df = test_df.loc[test_df[COL_EDGE].isin(self.rels_assoc_by_node_endpoint_type_dict.get(entity_type_to_predict).get(node_endpoint_to_predict))]
        return test_subset_df
    


def compute_metrics_over_triples_with_ent_node_endpoint(model_id, 
                                                        test_triples_w_node_degree_df, 
                                                        test_set_slug,
                                                        node_endpoint_type,                                                           
                                                        train_triples,
                                                        valid_triples,
                                                        entity_type_to_predict,
                                                        rels_assoc_by_node_endpoint_type_dict,
                                                        model_registry_cfg: ModelRegistryConfig,
):
    col_node_degree = f"{node_endpoint_type}_training_degree" # if predicting head, this should be the 'src_training_degree'
    evaluator = RankBasedEvaluator(filtered=True)
    model_base_path = model_registry_cfg.registered_model_paths.get(model_id)
    model = load_kge_model(model_base_path=model_base_path)
    print(f'loaded model from {str(model_base_path)}')
    additional_filter_triples = obtain_filtered_triples(test_type=test_set_slug,
                                                        train_triples=train_triples,
                                                        valid_triples=valid_triples
                                                       )
    
    test_triples_with_ent_node_endpoint_df = test_triples_w_node_degree_df.loc[test_triples_w_node_degree_df[COL_EDGE].isin(
        rels_assoc_by_node_endpoint_type_dict.get(entity_type_to_predict)[node_endpoint_type])]
    ent_degree_values = test_triples_with_ent_node_endpoint_df[col_node_degree].unique()
    result_dicts = []
    
    for degree_val in tqdm(ent_degree_values): 
        df_subset = test_triples_with_ent_node_endpoint_df.loc[test_triples_with_ent_node_endpoint_df[col_node_degree]==degree_val][[COL_SOURCE, COL_EDGE, COL_TARGET]]
        triples_subset = df_subset.values
        triples_subset = TriplesFactory.from_labeled_triples(triples_subset, 
                                                             relation_to_id=train_triples.relation_to_id, 
                                                             entity_to_id=train_triples.entity_to_id)
        if triples_subset.num_triples > 0:
            subset_result = evaluator.evaluate(model,
                                               triples_subset.mapped_triples, 
                                               additional_filter_triples=additional_filter_triples,
                                               use_tqdm=False)
            result_dicts.append({'ent_degree': degree_val, 'results': subset_result, 'relation': 'All', 'count': triples_subset.num_triples})
    return result_dicts

