from pykeen.models import ERModel

from bioblp.models.encoders import PropertyEncoderRepresentation


class BioBLP(ERModel):
    def __init__(self, *,
                 interaction_function: str,
                 entity_representations: PropertyEncoderRepresentation,
                 default_learning_rate: float,
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

        self.base_entity_representations = entity_representations
        self.default_learning_rate = default_learning_rate

    def get_grad_params(self):
        params = []

        added_modules = set()
        for encoder in self.base_entity_representations.type_id_to_encoder.values():
            for module in encoder.modules():
                lr = encoder.learning_rate or self.default_learning_rate
                params.append({'params': module.parameters(recurse=False),
                               'lr': lr})
                added_modules.add(module)

        for module in self.modules():
            if module not in added_modules:
                params.append({'params': module.parameters(recurse=False),
                               'lr': self.default_learning_rate})

        return params
