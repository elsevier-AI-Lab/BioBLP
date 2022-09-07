from pykeen.models import ERModel
from pykeen.nn.emb import EmbeddingSpecification
from pykeen.nn.modules import ComplExInteraction

from bioblp.models.encoders import PropertyEncoderRepresentation


class BioBLP(ERModel):
    def __init__(self, *, entity_representations: PropertyEncoderRepresentation,
                 embedding_dim, regularizer, **kwargs):
        dim = entity_representations.shape[0]
        relation_representations = EmbeddingSpecification(embedding_dim=dim)

        super().__init__(interaction=ComplExInteraction(),
                         entity_representations=entity_representations,
                         relation_representations=relation_representations,
                         **kwargs)
