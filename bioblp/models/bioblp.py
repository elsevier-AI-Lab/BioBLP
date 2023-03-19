import os.path as osp
from typing import Optional
from typing import Tuple

import pykeen.models
from pykeen.nn.representation import Embedding as PyKEmbedding
from pykeen.typing import InductiveMode
import torch

from bioblp.models.encoders import PropertyEncoderRepresentation


class BioBLP:
    def __init__(self, *,
                 entity_representations: PropertyEncoderRepresentation,
                 from_checkpoint: str = None,
                 **kwargs):
        self.from_checkpoint = from_checkpoint

        super().__init__(**kwargs)

        entity_embedding_lut = self.entity_representations[0]
        entity_embedding_lut: PyKEmbedding

        entity_representations.wrap_lookup_table(entity_embedding_lut)
        self.property_encoder = entity_representations

    def reset_parameters_(self):
        super().reset_parameters_()
        if self.from_checkpoint:
            checkpoint = torch.load(osp.join(self.from_checkpoint,
                                             'trained_model.pkl'),
                                    map_location='cpu')
            self.load_state_dict(checkpoint.state_dict(), strict=False)

    def score_hrt_and_negatives(self,
                                hrt_batch: torch.LongTensor,
                                num_negatives: int,
                                *, mode: Optional[InductiveMode] = None
                                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = hrt_batch.shape[0]

        h, r, t = self._get_representations(h=hrt_batch[:, 0],
                                            r=hrt_batch[:, 1],
                                            t=hrt_batch[:, 2], mode=mode)
        positive_scores = self.interaction.score_hrt(h=h, r=r, t=t)

        num_ents = batch_size * 2
        idx = torch.arange(num_ents).reshape(batch_size, 2)

        # For each row, sample entities, assigning 0 probability to entities
        # of the same row
        zeros = torch.zeros(batch_size, 2)
        head_weights = torch.ones(batch_size, num_ents, dtype=torch.float)
        head_weights.scatter_(1, idx, zeros)
        random_idx = head_weights.multinomial(num_negatives, replacement=True)
        random_idx = random_idx.t().flatten()

        # Select randomly the first or the second column
        row_selector = torch.arange(batch_size * num_negatives)
        col_selector = torch.randint(0, 2, [batch_size * num_negatives])

        # Fill the array of negative samples with the sampled random entities
        # at the right positions
        neg_idx = idx.repeat((num_negatives, 1))
        neg_idx[row_selector, col_selector] = random_idx
        # neg_idx = neg_idx.reshape(-1, batch_size, 2)
        # neg_idx.transpose_(0, 1)

        neg_embs = torch.stack([h, r], dim=1).view(batch_size * 2, -1)
        neg_embs = neg_embs[neg_idx.to(neg_embs.device)]
        h_neg, t_neg = neg_embs[:, 0], neg_embs[:, 1]

        r_neg_idx = torch.arange(batch_size).repeat(num_negatives)
        r_neg = r[r_neg_idx.to(r.device)]

        negative_scores = self.interaction.score_hrt(h=h_neg, r=r_neg, t=t_neg)
        negative_scores = negative_scores.reshape(batch_size, num_negatives)

        return positive_scores, negative_scores


class BioBLPTransE(BioBLP, pykeen.models.TransE):
    ...


class BioBLPComplEx(BioBLP, pykeen.models.ComplEx):
    ...


class BioBLPRotatE(BioBLP, pykeen.models.RotatE):
    ...


MODELS_DICT = {
    'transe': BioBLPTransE,
    'complex': BioBLPComplEx,
    'rotate': BioBLPRotatE
}


def get_model_class(model_name: str):
    if model_name in MODELS_DICT:
        return MODELS_DICT[model_name]
    else:
        raise ValueError(f'Unknown model f{model_name}')

