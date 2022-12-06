from pykeen.models import ERModel
from pykeen.nn.modules import (TransEInteraction, ComplExInteraction,
                               RotatEInteraction)

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
