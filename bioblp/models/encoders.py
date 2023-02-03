from functools import partial
from typing import Mapping, Optional, Tuple, Iterable

from pykeen.nn.representation import Embedding as PyKEmbedding
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from ..loaders.preprocessors import (TextEntityPropertyPreprocessor,
                                     MolecularFingerprintPreprocessor,
                                     PretrainedEmbeddingPreprocessor,
                                     MoleculeEmbeddingPreprocessor)


class PropertyEncoder(nn.Module):
    """An abstract class for encoders of entities with different properties"""
    def __init__(self, file_path: str = None, dim: int = None):
        super().__init__()
        self.file_path = file_path
        self.dim = dim

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def forward(self, data: Tensor, device: torch.device) -> Tensor:
        raise NotImplementedError


class LookupTableEncoder(PropertyEncoder):
    """A lookup table encoder for entities without properties."""
    def __init__(self, num_embeddings: int, dim: int):
        super().__init__(dim=dim)
        self.embeddings = nn.Embedding(num_embeddings, dim)

    def forward(self, indices: Tensor, device: torch.device) -> Tensor:
        return self.embeddings(indices.to(device))


class PretrainedLookupTableEncoder(PropertyEncoder):
    def __init__(self, file_path: str, dim: int):
        super().__init__(file_path, dim)

        data_dict = torch.load(file_path)

        num_entities = len(data_dict['identifiers'])
        in_dim = data_dict['embeddings'].shape[-1]

        self.embeddings = nn.Embedding(num_entities, in_dim)
        self.embeddings.weight.data = data_dict['embeddings']
        # TODO: dim could be higher than in_dim
        self.linear = nn.Linear(in_dim, dim)

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        preprocessor = PretrainedEmbeddingPreprocessor()
        return preprocessor.preprocess_file(self.file_path, entity_to_id)

    def forward(self, indices: Tensor, device: torch.device) -> Tensor:
        embs = self.embeddings(indices.to(device))
        embs = self.linear(embs)
        return embs


class MolecularFingerprintEncoder(PropertyEncoder):
    """Encoder of molecules described by a fingerprint"""
    def __init__(self, file_path: str, in_features: int, dim: int):
        super().__init__(file_path, dim)

        self.layers = nn.Sequential(nn.Linear(in_features, in_features // 2),
                                    nn.ReLU(),
                                    nn.Linear(in_features // 2, dim))

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        processor = MolecularFingerprintPreprocessor()
        data_tuple = processor.preprocess_file(self.file_path, entity_to_id)

        entities, data_idx, data = data_tuple

        return entities, data_idx, data

    def forward(self, data: Tensor, device: torch.device) -> Tensor:
        return self.layers(data.to(device))


class MoleculeEmbeddingEncoder(PropertyEncoder):
    def __init__(self, file_path: str, dim: int):
        super().__init__(file_path, dim)

        data_dict = torch.load(file_path)
        in_dim = next(iter(data_dict.values())).shape[-1]

        self.self_attention = nn.MultiheadAttention(in_dim,
                                                    num_heads=4,
                                                    batch_first=True)
        self.linear_hidden = nn.Linear(in_dim, in_dim)
        self.linear_out = nn.Linear(in_dim, dim)

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        processor = MoleculeEmbeddingPreprocessor()
        return processor.preprocess_file(self.file_path, entity_to_id)

    def forward(self, data: Tensor, device: torch.device) -> Tensor:
        # data: (batch_size, length, dim)
        x = data.to(device)

        # attention_mask is True for positions that need to be treated as
        # padding (ignored by self-attention). Padding has values of -10,000
        attention_mask = (x < -1_000)
        # To avoid any potential issues when actually computing self-attention,
        # we rewrite values from -10,000 to 0.0
        x[attention_mask] = 0.0
        attention_mask = attention_mask.any(dim=-1)

        x = self.self_attention(x, x, x, key_padding_mask=attention_mask)[0]
        x = torch.relu(self.linear_hidden(x[:, 0]))
        x = self.linear_out(x)

        return x


class TransformerTextEncoder(PropertyEncoder):
    """An encoder of entities with textual descriptions that uses BERT.
    Produces an embedding of an entity by passing the representation of
    the [CLS] symbol through a linear layer."""
    BASE_MODEL = 'dmis-lab/biobert-base-cased-v1.2'

    def __init__(self, file_path: str, dim: int):
        super().__init__(file_path, dim)
        self.encoder = AutoModel.from_pretrained(self.BASE_MODEL)

        encoder_hidden_size = self.encoder.config.hidden_size
        self.linear_out = nn.Linear(encoder_hidden_size, dim)

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        processor = TextEntityPropertyPreprocessor(tokenizer, max_length=64)
        data_tuple = processor.preprocess_file(self.file_path, entity_to_id)
        entities, data_idx, data = data_tuple

        return entities, data_idx, data

    def forward(self, tokens: Tensor, device: torch.device) -> Tensor:
        # Clip to maximum length in batch and move to device
        mask = (tokens > 0).float()
        max_length = mask.sum(dim=1).max().int().item()
        tokens = tokens[:, :max_length].to(device)
        mask = mask[:, :max_length].to(device)

        # Extract BERT representation of [CLS] token
        embeddings = self.encoder(tokens, mask)[0][:, 0]
        embeddings = self.linear_out(embeddings)

        return embeddings


class PropertyEncoderRepresentation(nn.Module):
    """A representation that maps entity IDs to an embedding that is a function
    of entity properties. Supports heterogeneous properties each with a
    potentially different encoder.
    """
    def __init__(self, dim: int, entity_to_id: Mapping[str, int],
                 encoders: Iterable[nn.Module]):
        num_entities = len(entity_to_id)

        super().__init__()

        self.dim = dim
        self.entity_types = torch.full([num_entities], fill_value=-1,
                                       dtype=torch.long)
        self.entity_data_idx = torch.zeros_like(self.entity_types)
        self.type_id_to_data = dict()
        self.type_id_to_encoder = dict()

        for type_id, encoder in enumerate(encoders):
            self.type_id_to_encoder[type_id] = encoder
            data_tuple = encoder.preprocess_properties(entity_to_id)
            entities, data_idx, data = data_tuple

            self.entity_types[entities] = type_id
            self.entity_data_idx[entities] = data_idx
            self.type_id_to_data[type_id] = data

        # Entities with no specified encoder are assigned a regular embedding
        self.unspecified_type_id = len(self.type_id_to_data)
        unspecified_mask = self.entity_types == -1
        num_unspecified_entities = unspecified_mask.sum().int().item()
        unspecified_index = torch.arange(num_unspecified_entities)

        self.entity_types[unspecified_mask] = self.unspecified_type_id
        self.entity_data_idx[unspecified_mask] = unspecified_index
        self.type_id_to_data[self.unspecified_type_id] = unspecified_index

        self.type_ids = torch.tensor(list(self.type_id_to_data.keys()),
                                     dtype=torch.long)

        for type_id, encoder in self.type_id_to_encoder.items():
            self.add_module(f'type_{type_id}_encoder', encoder)

        embeddings_buffer = torch.zeros([num_entities, dim], dtype=torch.float,
                                        requires_grad=False)
        self.register_buffer('embeddings_buffer', embeddings_buffer)

    def wrap_lookup_table(self, lookup_table: PyKEmbedding):
        lookup_table._plain_forward = partial(
            PropertyEncoderRepresentation.encode_entities,
            lookup_table=lookup_table,
            encoder_modules=self
        )

    @staticmethod
    def encode_entities(lookup_table: PyKEmbedding,
                        indices: Optional[torch.LongTensor],
                        encoder_modules: 'PropertyEncoderRepresentation'
                        ) -> Optional[torch.Tensor]:
        # This method is adapted from
        # pykeen.nn.representation.Embedding._plain_forward

        if indices is None:
            # Here we assume that if indices is None, we are not training,
            # so we pass the buffer of embeddings of all entities
            prefix_shape = (lookup_table.max_id,)
            x = encoder_modules.embeddings_buffer
        else:
            prefix_shape = indices.shape
            if encoder_modules.training:
                batch_size = indices.shape[0]
                device = indices.device
                x = torch.empty([batch_size, encoder_modules.dim],
                                dtype=torch.float,
                                device=device)

                # Sadly we have to move indices back to cpu to get property data
                indices = indices.cpu()
                entity_types = encoder_modules.entity_types[indices]
                type_assignments = entity_types.unsqueeze(-1) == encoder_modules.type_ids
                types_in_batch = torch.unique(entity_types).tolist()

                for t in types_in_batch:
                    entity_type_mask = type_assignments[:, t]
                    entities = indices[entity_type_mask]

                    if t != encoder_modules.unspecified_type_id:
                        type_t_data = encoder_modules.type_id_to_data[t]
                        data = type_t_data[encoder_modules.entity_data_idx[entities]]
                        encoder = encoder_modules.type_id_to_encoder[t]
                        embeddings = encoder(data, device=device)
                    else:
                        embeddings = lookup_table._embeddings(entities.to(device))

                    x[entity_type_mask] = embeddings
                    encoder_modules.embeddings_buffer[indices] = x.detach()
            else:
                x = encoder_modules.embeddings_buffer[indices]

        x = x.view(*prefix_shape, *lookup_table._shape)
        if lookup_table.is_complex:
            x = torch.view_as_complex(x)
        # verify that contiguity is preserved
        assert x.is_contiguous()
        return x
