from typing import Optional

from pykeen.models import ERModel
from pykeen.typing import InductiveMode
import torch

from bioblp.models.encoders import PropertyEncoderRepresentation


class BioBLP(ERModel):
    def __init__(self, *,
                 interaction_function: str,
                 entity_representations: PropertyEncoderRepresentation,
                 embedding_dim,
                 regularizer,
                 **kwargs):
        interaction_kwargs = dict()
        if interaction_function == 'transe':
            interaction_kwargs['p'] = 1

        super().__init__(interaction=interaction_function,
                         entity_representations=entity_representations,
                         relation_representations_kwargs={
                             'shape': entity_representations.shape},
                         interaction_kwargs=interaction_kwargs,
                         **kwargs)

    def score_hrt_and_negatives(self,
                                hrt_batch: torch.LongTensor,
                                num_negatives: int,
                                *, mode: Optional[InductiveMode] = None
                                ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
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
