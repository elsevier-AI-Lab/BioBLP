import os.path as osp

from pykeen.pipeline import pipeline
from pykeen.training import TrainingCallback
from pykeen.triples import TriplesFactory
from tap import Tap
import wandb

from bioblp.logging import get_logger


class Arguments(Tap):
    train_triples: str
    valid_triples: str
    test_triples: str

    model: str = 'complex'
    dimension: int = 256
    loss_fn: str = 'crossentropy'
    optimizer: str = 'adagrad'
    learning_rate: float = 1e-2
    regularizer: float = 1e-6
    num_epochs: int = 500
    batch_size: int = 512
    num_negatives: int = 128
    add_inverses: bool = False

    log_wandb: bool = False
    notes: str = None


class WBIDCallback(TrainingCallback):
    """A callback to get the wandb ID of the run before it gets closed.
    We use it to get a file name for the stored model."""
    id = None

    def post_train(self, *args, **kwargs):
        if wandb.run is not None:
            WBIDCallback.id = wandb.run.id


def run(args: Arguments):
    cli_args_dict = {f'cli_{k}': v for k, v in args.as_dict().items()}

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
                                       'batch_size': args.batch_size,
                                       'callbacks': WBIDCallback},
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
                          'config': cli_args_dict,
                          'offline': not args.log_wandb
                      }
                      )

    result.save_to_directory(osp.join('models', WBIDCallback.id))


if __name__ == '__main__':
    run(Arguments(explicit_bool=True).parse_args())
