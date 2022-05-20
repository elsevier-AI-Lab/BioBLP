from argparse import ArgumentParser
import os.path as osp
import logging
from datetime import datetime

from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss, BCEWithLogitsLoss
from torch.optim import Adagrad
import torch
import wandb

from bioblp.logging import get_logger
from bioblp.utils.triples import load_triples_array
from bioblp.utils.triples import DIR_PROCESSED

logger = get_logger(__name__)


TRANSE = 'transe'
COMPLEX = 'complex'
MODELS = {TRANSE, COMPLEX}
CMD_TRAIN = 'train'
CMD_SEARCH = 'search'


def parse_args():
    parser = ArgumentParser(description='Train a link prediction model')

    parser.add_argument('--command', choices={CMD_TRAIN, CMD_SEARCH},
                        default=CMD_TRAIN)
    parser.add_argument('--data_path', type=str,
                        help='Path containing the "processed" folder')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset for training. Assumes there are '
                             'corresponding train, validation and test splits '
                             'for it at data_path/processed/')

    parser.add_argument('--model', type=str, help='Link prediction model',
                        choices=MODELS)
    parser.add_argument('--dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin used in the margin ranking loss')
    parser.add_argument('--regularizer', type=float, default=1e-6,
                        help='Regularization coefficient')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--name', type=str, help='Experiment name used in logs')
    parser.add_argument('--notes', type=str, help='Notes to log with experiment')
    parser.add_argument('--offline', help='Add this option to disable wandb',
                        action='store_true')

    return parser.parse_args()


def get_default_model_kwargs(model: str, dim: int, margin: float):
    model_kwargs = dict(embedding_dim=dim)
    loss_kwargs = None
    if model == TRANSE:
        model_kwargs.update(dict(scoring_fct_norm=1, loss=MarginRankingLoss))
        loss_kwargs = dict(margin=margin)
    elif model == COMPLEX:
        model_kwargs.update(dict(loss=BCEWithLogitsLoss))
    else:
        raise ValueError(f'Model {model} unknown, choose one of {MODELS}')

    return model_kwargs, loss_kwargs


def run(args):
    processed_path = osp.join(args.data_path, DIR_PROCESSED)

    train_triples = load_triples_array(
        osp.join(processed_path, f'{args.dataset}-train.tsv'))
    valid_triples = load_triples_array(
        osp.join(processed_path, f'{args.dataset}-valid.tsv'))
    test_triples = load_triples_array(
        osp.join(processed_path, f'{args.dataset}-test.tsv'))

    training = TriplesFactory.from_labeled_triples(train_triples)
    validation = TriplesFactory.from_labeled_triples(valid_triples,
                                                     entity_to_id=training.entity_to_id,
                                                     relation_to_id=training.relation_to_id)
    testing = TriplesFactory.from_labeled_triples(test_triples,
                                                  entity_to_id=training.entity_to_id,
                                                  relation_to_id=training.relation_to_id)

    logger.info(f'Starting training on {args.dataset}')
    logger.info(f'Number of triples: '
                f'{training.num_triples:,} (training) '
                f'{validation.num_triples:,} (validation) '
                f'{testing.num_triples:,} (testing)')
    logger.info(f'{training.num_entities} entities and '
                f'{training.num_relations} relations')

    if args.name is not None:
        experiment = f'{args.model}-{args.command}-{args.name}'
    else:
        experiment = f'{args.model}-{args.command}'

    if args.command == CMD_SEARCH:
        model_kwargs = dict(embedding_dim=128)
        loss_ranges = dict()
        if args.model == COMPLEX:
            loss = BCEWithLogitsLoss
        elif args.model == TRANSE:
            model_kwargs.update(dict(scoring_fct_norm=1))
            loss = MarginRankingLoss
            loss_ranges.update(dict(margin=dict(type=int, low=1, high=10, q=1)))
        else:
            raise ValueError(f'Unkown model {args.model}, choose one of {MODELS}')

        date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        hpo_directory = osp.join('results', f'{date_str}-{experiment}')

        result = hpo_pipeline(n_trials=args.num_trials,
                              training=training,
                              validation=validation,
                              testing=testing,
                              model=args.model,
                              model_kwargs=model_kwargs,
                              loss=loss,
                              loss_kwargs_ranges=loss_ranges,
                              regularizer='LpRegularizer',
                              regularizer_kwargs_ranges=dict(
                                  weight=dict(type=float,
                                              low=1e-10,
                                              high=1e-6,
                                              scale='log')),
                              optimizer=Adagrad,
                              optimizer_kwargs=dict(lr=0.1),
                              training_loop='slcwa',
                              negative_sampler='basic',
                              negative_sampler_kwargs=dict(
                                  num_negs_per_pos=128),
                              training_kwargs=dict(num_epochs=args.num_epochs),
                              training_kwargs_ranges=dict(
                                  batch_size=dict(type=int,
                                                  low=100,
                                                  high=500,
                                                  q=100)),
                              stopper='early',
                              stopper_kwargs=dict(
                                  metric='both.pessimistic.inverse_harmonic_mean_rank',
                                  frequency=10,
                                  patience=5,
                                  relative_delta=0.0001,
                                  larger_is_better=True),
                              evaluator_kwargs=dict(filtered=True),
                              result_tracker='wandb',
                              result_tracker_kwargs=dict(
                                  entity='discoverylab',
                                  project='bioblp',
                                  experiment=experiment,
                                  notes=args.notes,
                                  reinit=True,
                                  offline=args.offline),
                              save_model_directory=hpo_directory
                              )
        result.save_to_directory(osp.join(hpo_directory, 'study'))

        # Run evaluation pipeline (with 0 training epochs) with best model
        best_model = torch.load(osp.join(hpo_directory,
                                         str(result.study.best_trial.number),
                                         'trained_model.pkl'))
        pipeline(model=best_model,
                 training=training, validation=validation, testing=testing,
                 training_kwargs=dict(num_epochs=0),
                 result_tracker='wandb',
                 result_tracker_kwargs=dict(
                     entity='discoverylab',
                     project='bioblp',
                     experiment=f'{experiment}-best-test',
                     notes=args.notes,
                     reinit=True,
                     offline=args.offline),
                 )
    else:
        model_kwargs, loss_kwargs = get_default_model_kwargs(args.model,
                                                             args.dim,
                                                             args.margin)
        result = pipeline(training=training,
                          validation=validation,
                          testing=testing,
                          model=args.model,
                          model_kwargs=model_kwargs,
                          loss_kwargs=loss_kwargs,
                          regularizer='LpRegularizer',
                          regularizer_kwargs=dict(weight=args.regularizer),
                          optimizer=Adagrad,
                          optimizer_kwargs=dict(lr=0.1),
                          training_kwargs=dict(num_epochs=args.num_epochs,
                                               batch_size=args.batch_size),
                          negative_sampler='basic',
                          negative_sampler_kwargs=dict(num_negs_per_pos=128),
                          stopper='early',
                          stopper_kwargs=dict(
                              metric='both.pessimistic.inverse_harmonic_mean_rank',
                              frequency=10,
                              patience=5,
                              relative_delta=0.0001,
                              larger_is_better=True),
                          evaluator_kwargs=dict(filtered=True),
                          result_tracker='wandb',
                          result_tracker_kwargs=dict(entity='discoverylab',
                                                     project='bioblp',
                                                     experiment=experiment,
                                                     notes=args.notes,
                                                     offline=args.offline))

        result.save_to_directory(osp.join('results', wandb.run.id))


if __name__ == '__main__':
    run(parse_args())
