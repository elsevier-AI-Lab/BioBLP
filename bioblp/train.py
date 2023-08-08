import os
import os.path as osp
import string
import random

from pykeen.pipeline import pipeline
from pykeen.training import TrainingCallback
from pykeen.triples import TriplesFactory

from tap import Tap
from transformers import get_linear_schedule_with_warmup
import wandb

from bioblp.logger import get_logger
import bioblp.models as models
from bioblp.utils.bioblp_utils import build_encoders
from bioblp.utils.training import InBatchNegativesTraining


class Arguments(Tap):
    train_triples: str
    valid_triples: str
    test_triples: str

    protein_data: str = None
    molecule_data: str = None
    text_data: str = None

    model: str = 'complex'
    dimension: int = 256
    loss_fn: str = 'crossentropy'
    loss_margin: float = 1.0
    optimizer: str = 'adagrad'
    learning_rate: float = 1e-2
    freeze_pretrained_embeddings: bool = False
    warmup_fraction: float = None
    regularizer: float = 1e-6
    num_epochs: int = 100
    batch_size: int = 1024
    eval_batch_size: int = 16
    eval_every: int = 10
    num_negatives: int = 512
    in_batch_negatives: bool = False
    add_inverses: bool = False
    early_stopper: str = 'both.realistic.inverse_harmonic_mean_rank'
    from_checkpoint: str = None

    search_train_batch_size: bool = False
    search_eval_batch_size: bool = False
    log_wandb: bool = False
    log_mlflow: bool = False
    notes: str = None


class BioBLPCallback(TrainingCallback):
    """A callback to get the wandb ID of the run before it gets closed.
    We use it to get a file name for the stored model."""
    id = None
    scheduler = None

    def __init__(self, num_training_steps, warmup_fraction):
        super().__init__()
        self.use_scheduler = warmup_fraction is not None
        if self.use_scheduler:
            self.num_training_steps = num_training_steps
            self.num_warmup_steps = int(
                self.num_training_steps * warmup_fraction)

    def post_epoch(self, *args, **kwargs):
        if wandb.run is not None and BioBLPCallback.id is None:
            BioBLPCallback.id = wandb.run.id
        elif wandb.run is None and BioBLPCallback.id is None:
            BioBLPCallback.id = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=8))

    def pre_step(self, **kwargs):
        if not self.use_scheduler:
            return

        if self.scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                self.num_warmup_steps,
                self.num_training_steps
            )
        else:
            self.scheduler.step()


def run(args: Arguments):
    cli_args_dict = {f'cli_{k}': v for k, v in args.as_dict().items()}
    if args.search_train_batch_size:
        args.batch_size = None
    if args.search_eval_batch_size:
        args.eval_batch_size = None

    logger = get_logger()
    logger.info('Loading triples...')

    entity_to_id = relation_to_id = None
    if args.from_checkpoint:
        checkpoint_triples = TriplesFactory.from_path_binary(
            osp.join(args.from_checkpoint, 'training_triples')
        )
        entity_to_id = checkpoint_triples.entity_to_id
        relation_to_id = checkpoint_triples.relation_to_id

    training = TriplesFactory.from_path(
        args.train_triples,
        create_inverse_triples=args.add_inverses,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    validation = TriplesFactory.from_path(args.valid_triples,
                                          entity_to_id=training.entity_to_id,
                                          relation_to_id=training.relation_to_id)
    testing = TriplesFactory.from_path(args.test_triples,
                                       entity_to_id=training.entity_to_id,
                                       relation_to_id=training.relation_to_id)

    logger.info(f'Loaded graph with {training.num_entities:,} entities')
    logger.info(f'{training.num_triples:,} training triples')
    logger.info(f'{validation.num_triples:,} validation triples')
    logger.info(f'{testing.num_triples:,} test triples')

    loss_kwargs = None
    if args.loss_fn in {'nssa', 'marginranking'}:
        loss_kwargs = {'margin': args.loss_margin}
    model = args.model
    model_kwargs = {'embedding_dim': args.dimension, 'loss': args.loss_fn}

    if any((args.protein_data, args.molecule_data, args.text_data)):
        model = models.get_model_class(args.model)
        dimension = args.dimension
        if args.model in ('complex', 'rotate'):
            dimension *= 2

        freeze_pretrained_embeddings = args.freeze_pretrained_embeddings
        encoders = build_encoders(dimension,
                                  training.entity_to_id,
                                  args.protein_data,
                                  args.molecule_data,
                                  args.text_data,
                                  freeze_pretrained_embeddings)
        model_kwargs['entity_representations'] = encoders

        if args.from_checkpoint:
            model_kwargs['from_checkpoint'] = args.from_checkpoint

    if args.warmup_fraction:
        if args.batch_size is None:
            raise ValueError('Batch size is needed to apply learning rate'
                             ' warmup.')
        num_steps = (training.num_triples // args.batch_size) * args.num_epochs
    else:
        num_steps = None

    training_loop = InBatchNegativesTraining if args.in_batch_negatives else None

    # set trackers

    # default tracker is wandb
    result_tracker = "wandb"
    result_tracker_kwargs = {
        'entity': 'discoverylab',
        'project': 'bioblp',
        'notes': args.notes,
        'config': cli_args_dict,
        'offline': not args.log_wandb
    }
    if args.log_mlflow and not args.log_wandb:
        result_tracker = "mlflow"

        result_tracker_kwargs = {
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", None),
            "experiment_name": "biograph-test",
            "tags": {
                "notes": args.notes
            }
        }
    elif args.log_mlflow and args.log_wandb:
        logger.info(
            f"Logging to WandB AND MLFlow not supported, deferring to default WandB...")

    result = pipeline(training=training,
                      validation=validation,
                      testing=testing,
                      model=model,
                      model_kwargs=model_kwargs,
                      loss_kwargs=loss_kwargs,
                      optimizer=args.optimizer,
                      optimizer_kwargs={'lr': args.learning_rate},
                      regularizer='LpRegularizer',
                      regularizer_kwargs={'weight': args.regularizer},
                      training_kwargs={'num_epochs': args.num_epochs,
                                       'batch_size': args.batch_size,
                                       'callbacks': BioBLPCallback,
                                       'callback_kwargs': {
                                           'num_training_steps': num_steps,
                                           'warmup_fraction': args.warmup_fraction
                                       }},
                      training_loop=training_loop,
                      negative_sampler='basic',
                      negative_sampler_kwargs={
                          'num_negs_per_pos': args.num_negatives
                      },
                      stopper='early',
                      stopper_kwargs={
                          'evaluation_batch_size': args.eval_batch_size,
                          'metric': args.early_stopper,
                          'frequency': args.eval_every,
                          'patience': 5,
                          'relative_delta': 0.0001,
                          'larger_is_better': True
                      },
                      evaluator_kwargs={'batch_size': args.eval_batch_size},
                      result_tracker=result_tracker,
                      result_tracker_kwargs=result_tracker_kwargs
                      )

    result.save_to_directory(osp.join('models', BioBLPCallback.id))


if __name__ == '__main__':
    run(Arguments(explicit_bool=True).parse_args())
