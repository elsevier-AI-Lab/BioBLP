import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import wandb

###### Define CONSTANTS ######
TEST = "test"
VALID = "valid"
TRAIN = "train"
TEST_RESTRICTED_DIS = 'TEST_RESTRICTED_DIS'
TEST_EXCLUDING_DIS = 'TEST_EXCLUDING_DIS'
VALID_RESTRICTED_DIS = 'VALID_RESTRICTED_DIS'

COL_SOURCE = "src"
COL_TARGET = "tgt"
COL_EDGE = "rel"
COL_NODE = "ent"
DEGREE = "degree"
IN_DEGREE = "in_degree"
OUT_DEGREE = "out_degree"

# WandB stuff
WANDB_ENTITY_DISCOVERYLAB = "discoverylab"


# pykeen evaluation parameters
EVAL_NODE_BOTH = "both"
EVAL_NODE_HEAD = "head"
EVAL_NODE_TAIL = "tail"
EVAL_METRIC_REALISTIC = "realistic"
EVAL_METRICS_SHORTLIST = ["arithmetic_mean_rank", "adjusted_arithmetic_mean_rank", 
                          "inverse_harmonic_mean_rank", "hits_at_1",
                          "hits_at_3", "hits_at_5", "hits_at_10"] 


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
                                 wandb_entity_name:str=WANDB_ENTITY_DISCOVERYLAB,
                                 **eval_kwargs):
    tags = {"model_name": model_name,
            "study_name": study_name,
            "test_set_type": test_set_slug}
    metrics = eval_func(**eval_kwargs)
    run = setup_wandb_result_tracker(**tags, project_name=wandb_project_name, wandb_entity=wandb_entity_name)
    run.log(metrics)
    return metrics


##### evaluation logic ##### 


        

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
                      triple_endpoint="both",
                      metric_type = "realistic"):
    
    metrics_shortlist=["arithmetic_mean_rank", "adjusted_arithmetic_mean_rank", 
                       "inverse_harmonic_mean_rank", "hits_at_1",
                       "hits_at_3", "hits_at_5", "hits_at_10"]
    if not relation:
        relation= 'All'
    results = results.to_dict()
    results_dict = {relation: {
        'Count': relation_count,
        **{metric: results[triple_endpoint][metric_type][metric] for metric in metrics_shortlist}
    }}

    return results_dict


##### test set processing stuff #####

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
