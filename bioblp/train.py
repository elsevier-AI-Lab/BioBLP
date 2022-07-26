import os.path as osp

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from tap import Tap
import wandb

from bioblp.logging import get_logger

logger = get_logger(__name__)


class Arguments(Tap):
    training_triples: str
    valid_triples: str
    test_triples: str

    model: str = 'complex'
    dimension: int = 256
    loss_fn: str = 'crossentropy'
    optimizer: str = 'adagrad'
    learning_rate: float = 1e-2
    regularizer: float = 1e-6
    num_epochs: int = 100
    batch_size: int = 512
    num_negatives: int = 128
    add_inverses: bool = False

    log_wandb: bool = False
    notes: str = None


def run(args: Arguments):
    logger.info('Loading triples...')
    training = TriplesFactory.from_path(
        args.training_triples,
        create_inverse_triples=args.add_inverses
    )
    validation = TriplesFactory.from_path(args.valid_triples)
    testing = TriplesFactory.from_path(args.test_triples)

    logger.info(f'Loaded graph with {training.num_entities:,} entities')
    logger.info(f'{training.num_triples:,} training triples')
    logger.info(f'{validation.num_triples:,} validation triples')
    logger.info(f'{testing.num_triples:,} test triples')

    result = pipeline(training=training,
                      validation=validation,
                      testing=testing,
                      model=args.model,
                      model_kwargs={'embedding_dim': args.dimension,
                                    'loss': args.loss_fn},
                      optimizer=args.optimizer,
                      optimizer_kwargs={'lr': args.learning_rate},
                      regularizer='LpRegularizer',
                      regularizer_kwargs={'weight': args.regularizer},
                      training_kwargs={'num_epochs': args.num_epochs,
                                       'batch_size': args.batch_size},
                      negative_sampler='basic',
                      negative_sampler_kwargs={
                          'num_negs_per_pos': args.num_negatives
                      },
                      stopper='early',
                      stopper_kwargs={
                          'metric': 'both.realistic.inverse_harmonic_mean_rank',
                          'frequency': 10,
                          'patience': 5,
                          'relative_delta': 0.0001,
                          'larger_is_better': True
                      },
                      result_tracker='wandb',
                      result_tracker_kwargs={
                          'entity': 'discoverylab',
                          'project': 'bioblp',
                          'notes': args.notes,
                          'offline': not args.log_wandb
                      }
                      )

    result.save_to_directory(osp.join('models', wandb.run.id))


run(Arguments(explicit_bool=True).parse_args())
