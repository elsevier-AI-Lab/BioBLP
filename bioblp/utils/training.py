from typing import Optional, Union

from pykeen.training.slcwa import SLCWATrainingLoop
from pykeen.models.base import Model
from pykeen.losses import Loss
from pykeen.typing import InductiveMode
from pykeen.triples.instances import SLCWABatch
import torch

from bioblp.models import BioBLP


class InBatchNegativesTraining(SLCWATrainingLoop):
    @staticmethod
    def _process_batch_static(
            model: Union[BioBLP, Model],
            loss: Loss,
            mode: Optional[InductiveMode],
            batch: SLCWABatch,
            start: Optional[int],
            stop: Optional[int],
            label_smoothing: float = 0.0,
            slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError(
                "Slicing is not possible for sLCWA training loops.")

        positive_batch, negative_batch, positive_filter = batch
        positive_batch = positive_batch[start:stop].to(device=model.device)

        positive_scores, negative_scores = model.score_hrt_and_negatives(
            positive_batch,
            num_negatives=negative_batch.shape[1],
            mode=mode
        )

        return (
                loss.process_slcwa_scores(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    label_smoothing=label_smoothing,
                    batch_filter=positive_filter,
                    num_entities=model._get_entity_len(mode=mode),
                )
                + model.collect_regularization_term()
        )