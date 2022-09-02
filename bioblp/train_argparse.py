import os.path as osp
from pathlib import Path
from pykeen.pipeline import pipeline
from pykeen.training import TrainingCallback
from pykeen.triples import TriplesFactory
from dataclasses import dataclass, asdict
# from tap import Tap
from argparse import ArgumentParser
import wandb
import toml

from bioblp.logging import get_logger

@dataclass
class Arguments:
    #data_splits_path: str
    #dataset_name: str
    train_triples: str
    valid_triples: str
    test_triples: str

    model: str = 'complex'
    dimension: int = 256
    loss_fn: str = 'crossentropy'
    loss_margin: float = 1.0
    optimizer: str = 'adagrad'
    learning_rate: float = 1e-2
    regularizer: float = 1e-6
    num_epochs: int = 100
    batch_size: int = 1024
    eval_batch_size: int = 16
    num_negatives: int = 512
    add_inverses: bool = False
    early_stopper: str = 'both.realistic.inverse_harmonic_mean_rank'

    search_train_batch_size: bool = False
    search_eval_batch_size: bool = False
    log_wandb: bool = False
    notes: str = None


class WBIDCallback(TrainingCallback):
    """A callback to get the wandb ID of the run before it gets closed.
    We use it to get a file name for the stored model."""
    id = None

    def post_train(self, *args, **kwargs):
        if wandb.run is not None:
            WBIDCallback.id = wandb.run.id

            
def load_toml(toml_path: str) -> dict:
    toml_path = Path(toml_path)
    config = {}
    with open(toml_path, "r") as f:
        config = toml.load(f)

    return config


def run(args: Arguments):
    cli_args_dict = {f'cli_{k}': v for k, v in asdict(args).items()}
    if args.search_train_batch_size:
        args.batch_size = None
    if args.search_eval_batch_size:
        args.eval_batch_size = None

    logger = get_logger()
    logger.info('Loading triples...')

    training = TriplesFactory.from_path(
        args.train_triples,
        create_inverse_triples=args.add_inverses
    )
    validation = TriplesFactory.from_path(args.valid_triples)
    testing = TriplesFactory.from_path(args.test_triples)

    logger.info(f'Loaded graph with {training.num_entities:,} entities')
    logger.info(f'{training.num_triples:,} training triples')
    logger.info(f'{validation.num_triples:,} validation triples')
    logger.info(f'{testing.num_triples:,} test triples')

    loss_kwargs = None
    if args.loss_fn in {'nssa', 'marginranking'}:
        loss_kwargs = {'margin': args.loss_margin}

    result = pipeline(training=training,
                      validation=validation,
                      testing=testing,
                      model=args.model,
                      model_kwargs={'embedding_dim': args.dimension,
                                    'loss': args.loss_fn},
                      loss_kwargs=loss_kwargs,
                      optimizer=args.optimizer,
                      optimizer_kwargs={'lr': args.learning_rate},
                      regularizer='LpRegularizer',
                      #regularizer_kwargs={'weight': args.regularizer},
                      training_kwargs={'num_epochs': args.num_epochs,
                                       'batch_size': args.batch_size,
                                       'callbacks': WBIDCallback},
                      negative_sampler='basic',
                      negative_sampler_kwargs={
                          'num_negs_per_pos': args.num_negatives
                      },
                      stopper='early',
                      stopper_kwargs={
                          'evaluation_batch_size': args.eval_batch_size,
                          'metric': args.early_stopper,
                          'frequency': 10,
                          'patience': 5,
                          'relative_delta': 0.0001,
                          'larger_is_better': True
                      },
                      evaluator_kwargs={'batch_size': args.eval_batch_size},
                      result_tracker='wandb',
                      result_tracker_kwargs={
                          'entity': 'discoverylab',
                          'project': 'bioblp',
                          'notes': args.notes,
                          'config': cli_args_dict,
                          'offline': not args.log_wandb
                      }
                      )
    
    result.save_to_directory(osp.join('models', WBIDCallback.id))


if __name__ == '__main__':
    parser = ArgumentParser(description="Model training routing")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment toml file")
    #parser.add_argument('--out_path', type=str,
    #                    help='Path to write models output')
    
    args = parser.parse_args()
    conf = load_toml(args.conf)
    args = Arguments(**conf)
    run(args)
    #run(Arguments(explicit_bool=True).parse_args())
