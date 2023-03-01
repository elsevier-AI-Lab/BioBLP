import pandas as pd
from pykeen.triples import TriplesFactory


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


### test sets

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
    
    
def split_test_triples_conditioned_on_ent_property(train_triples: TriplesFactory, 
                                                   typed_ent_set: set,
                                                   typed_ent_with_prop_set: set,
                                                   test_triples: TriplesFactory):
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
