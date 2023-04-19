from abc import ABC
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import seaborn as sns
from tqdm import tqdm
import wandb
import time


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
    