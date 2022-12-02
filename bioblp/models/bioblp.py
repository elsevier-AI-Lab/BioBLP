from pykeen.models import ERModel
from pykeen.nn.modules import ComplExInteraction

from bioblp.models.encoders import PropertyEncoderRepresentation


class BioBLP(ERModel):
    def __init__(self, *, entity_representations: PropertyEncoderRepresentation,
                 embedding_dim, regularizer, **kwargs):

        super().__init__(interaction=ComplExInteraction(),
                         entity_representations=entity_representations,
                         relation_representations_kwargs={'shape': entity_representations.shape},
                         **kwargs)
