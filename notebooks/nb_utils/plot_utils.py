from abc import ABC
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm
import wandb
import time


from .eval_utils import EVAL_NODE_HEAD
from .eval_utils import COL_SOURCE, COL_TARGET
from .eval_utils import INVERSE_HARMONIC_MEAN_RANK
from .eval_utils import EVAL_METRICS_SHORTLIST
from .eval_utils import get_unique_endpoint_entities_in_testset_of_given_degree



DEFAULT_EVAL_IMG_DIR = "./data/imgs/"

#### Plotting utils

def extract_relevant_metrics(metric_name, results_by_node_degree_dict):
    degrees = list(results_by_node_degree_dict.keys())
    metrics = []
    for deg, metric in results_by_node_degree_dict.items():
        metrics.append(metric[metric_name])
    return degrees, metrics
    
    
def plot_metric_vs_degree_scatterplot_single_model(model_id,
                                                   metric_name='hits_at_10', 
                                                   results_by_node_degree_dict=None, 
                                                   figsize=(7,7),
                                                   entity_type_examined=None,  
                                                   eval_out_dir=DEFAULT_EVAL_IMG_DIR
                                                  ):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    eval_out_dir = Path(eval_out_dir)
    eval_out_dir.mkdir(exist_ok=True, parents=True)
    degrees, metrics = extract_relevant_metrics(metric_name, results_by_node_degree_dict)   
    plot, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(ax = ax, x=degrees, y=metrics)
    plt.xscale('log')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} predicting {entity_type_examined} nodes Vs. {entity_type_examined} node degree | Model: {model_id}')
    plt.savefig(eval_out_dir.joinpath(f'{timestr}-{metric_name}_node_degree_analysis-{model_id}'))
    plt.show()
 
    
    
def plot_metric_vs_degree_scatterplot_multi_models(metric_name='hits_at_10', 
                                                   results_by_node_degree_dicts_combined=None,
                                                   entity_type_examined=None,                                                                                         eval_out_dir=DEFAULT_EVAL_IMG_DIR
                                                  ):
    result_dfs = {}
    timestr = time.strftime("%Y%m%d-%H%M%S")
    eval_out_dir = Path(eval_out_dir)
    eval_out_dir.mkdir(exist_ok=True, parents=True)
    for model_id, result_dict in results_by_node_degree_dicts_combined.items():
        degrees, metrics = extract_relevant_metrics(metric_name, result_dict)
        result_dfs[model_id] = pd.DataFrame(np.column_stack([degrees, metrics]), columns=['degree', 'metrics'], dtype=float)
    model_ids = result_dfs.keys()
    if len(model_ids)>2:
        raise ValueError("This function currently handles results from 2 models. Amend logic to handle >2 models")

    concatenated = pd.concat([result_dfs[model_id].assign(dataset=f'{model_id}') for model_id in model_ids])
    plot, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(ax=ax, data=concatenated, x='degree', y='metrics', hue='dataset')
    plt.xscale('log')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} predicting {entity_type_examined} nodes Vs. {entity_type_examined} node degree | Model: {list(model_ids)}')
    plt.savefig(eval_out_dir.joinpath(f"{timestr}-{metric_name}_node_degree_analysis-{'-'.join(list(model_ids))}"))

    #return degrees, metrics

    
    
    
def merge_and_plot_node_degree_analysis_multimodel_lp_eval_diff(bioblp_model_id=None,
                                                                bioblp_eval_results=None,
                                                                rotate_model_id=None,
                                                                rotate_eval_results=None,
                                                                node_endpoint_type_for_entity_w_attribute = None,
                                                                eval_on_node_endpoint=EVAL_NODE_HEAD,
                                                            test_triples=None,
                                                                entity_type_w_attr_encoded=None,
                                                                metric_name=INVERSE_HARMONIC_MEAN_RANK
                                                               ):  
    merged_results_df = prep_data_for_node_degree_lp_multi_model_diff_plot(bioblp_model_id=bioblp_model_id,
                                                                           bioblp_eval_results=bioblp_eval_results,
                                                                           rotate_model_id=rotate_model_id,
                                                                           rotate_eval_results=rotate_eval_results,
                                                                           node_endpoint_type=node_endpoint_type_for_entity_w_attribute,
                                                                           eval_on_node_endpoint=eval_on_node_endpoint,
                                                                           test_triples=test_triples
                                                                          ) 
    
    plot_node_degree_analysis_multimodel_lp_eval_diff(merged_results_df=merged_results_df,
                                                      entity_type_w_attr_encoded=entity_type_w_attr_encoded,
                                                      bioblp_model_id=bioblp_model_id, 
                                                      rotate_model_id=rotate_model_id,
                                                      eval_on_node_endpoint=eval_on_node_endpoint,
                                                      node_endpoint_type_for_entity_w_attribute=node_endpoint_type_for_entity_w_attribute,                                                          metric_name=metric_name)
    
    return merged_results_df


def _merge_results_df_multi_models(rotate_df, bioblp_df, 
                     test_triples_subset=None, 
                     node_endpoint_type=None):
    merged_df = rotate_df.merge(bioblp_df, how='inner', on='degree', suffixes=['_rotate', '_bioblp'])
    merged_df.drop(columns=['Relation_bioblp', 'Relation_rotate', 'Count_bioblp'], inplace=True)
    merged_df = merged_df.astype({'degree':'float'})
    merged_df = merged_df.rename(columns={"Count_rotate":"count"})
    
    merged_df['num_unique_ents'] = merged_df['degree'].apply(
    lambda x: len(get_unique_endpoint_entities_in_testset_of_given_degree(test_triples_subset, 
                                                                          degree=x, 
                                                                          node_endpoint_type=node_endpoint_type)))
    return merged_df


def prep_data_for_node_degree_lp_multi_model_diff_plot(bioblp_model_id, bioblp_eval_results,
                                                       rotate_model_id, rotate_eval_results,
                                                       node_endpoint_type=COL_SOURCE,
                                                       eval_on_node_endpoint=EVAL_NODE_HEAD,
                                                       test_triples=None
                                                       ):
    rotate_df = pd.DataFrame.from_dict(rotate_eval_results[eval_on_node_endpoint], orient='index')
    #rotate_df = pd.DataFrame.from_dict(rotate_eval_results, orient='index')
    rotate_df = rotate_df.reset_index().rename(columns={'index':'degree'})
    rotate_df = rotate_df.astype({'degree':'float'})

    bioblp_df = pd.DataFrame.from_dict(bioblp_eval_results[eval_on_node_endpoint], orient='index')
    #bioblp_df = pd.DataFrame.from_dict(bioblp_eval_results, orient='index')
    bioblp_df = bioblp_df.reset_index().rename(columns={'index':'degree'})
    bioblp_df = bioblp_df.astype({'degree':'float'})

    merged_df = _merge_results_df_multi_models(rotate_df=rotate_df, 
                                               bioblp_df=bioblp_df, 
                                               test_triples_subset=test_triples,
                                               node_endpoint_type=node_endpoint_type)

    for metric in EVAL_METRICS_SHORTLIST:
        merged_df[metric] = merged_df[f'{metric}_rotate'] - merged_df[f'{metric}_bioblp']   
    return merged_df


def plot_node_degree_analysis_multimodel_lp_eval_diff(merged_results_df,
                                                      entity_type_w_attr_encoded,
                                                      bioblp_model_id, 
                                                      rotate_model_id,
                                                      node_endpoint_type_for_entity_w_attribute=COL_SOURCE,                                                                                                       
                                                      eval_on_node_endpoint=EVAL_NODE_HEAD,
                                                      metric_name=INVERSE_HARMONIC_MEAN_RANK):
    
    degrees, metrics, counts, num_unique_ents = merged_results_df['degree'], merged_results_df[metric_name], merged_results_df['count'], merged_results_df['num_unique_ents']
    plot, ax = plt.subplots(figsize=(10,10))
    sns.color_palette("flare", as_cmap=True)
    s_square = [n*n for n in num_unique_ents] 
    plot_ = sns.scatterplot(ax = ax, x=degrees, y=metrics, size=num_unique_ents, sizes=(10,200))
    plt.xscale('log')
    plt.ylabel(metric_name)
    plt.xlabel(f'Training degree for {entity_type_w_attr_encoded} entity being predicted')
    #ax.axis([1, 10000, 1, 1000000])
    #ax.loglog()
    plt.xticks(degrees, rotation='vertical')

    plot_.xaxis.set_major_locator(ticker.IndexLocator(100, 0))

    plt.title(f'(avg_{metric_name}_{rotate_model_id}) - (avg_{metric_name}_{bioblp_model_id}) \n Difference b/w both models in {metric_name} when'\
              'predicting {entity_type_w_attr_encoded} as {eval_on_node_type} node, Vs. {entity_type_w_attr_encoded} node degree')
    #plt.savefig(f'data/imgs/{metric_name}_node_degree_analysis-{bioblp_model_id}-{rotate_model_id}')
    #plt.savefig(f"data/imgs/{metric_name}_node_degree_analysis-{bioblp_model_id}-{rotate_model_id}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


