import pdb
from typing import Mapping, Optional, Tuple, Iterable

from pykeen.nn import Representation
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from ..loaders.preprocessors import (TextEntityPropertyPreprocessor,
                                     MolecularFingerprintPreprocessor,
                                     PretrainedEmbeddingPreprocessor)


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


class TransformerTextEncoder(PropertyEncoder):
    """An encoder of entities with textual descriptions that uses BERT.
    Produces an embedding of an entity by passing the representation of
    the [CLS] symbol through a linear layer."""
    BASE_MODEL = 'allenai/scibert_scivocab_uncased'

    def __init__(self, file_path: str, dim: int):
        super().__init__(file_path, dim)
        self.encoder = AutoModel.from_pretrained(self.BASE_MODEL)

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        encoder_hidden_size = self.encoder.config.hidden_size
        self.linear_out = nn.Linear(encoder_hidden_size, dim)

    def preprocess_properties(self,
                              entity_to_id: Mapping[str, int]
                              ) -> Tuple[Tensor, Tensor, Tensor]:
        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        processor = TextEntityPropertyPreprocessor(tokenizer, max_length=32)
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


class PropertyEncoderRepresentation(Representation):
    """A representation that maps entity IDs to an embedding that is a function
    of entity properties. Supports heterogeneous properties each with a
    potentially different encoder.
    """
    def __init__(self, dim: int, entity_to_id: Mapping[str, int],
                 encoders: Iterable[PropertyEncoder]):
        num_entities = len(entity_to_id)

        super().__init__(max_id=num_entities, shape=(dim,))

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
        unspecified_type_id = len(self.type_id_to_data)
        unspecified_mask = self.entity_types == -1
        num_unspecified_entities = unspecified_mask.sum().int().item()
        unspecified_index = torch.arange(num_unspecified_entities)

        lookup_encoder = LookupTableEncoder(num_unspecified_entities, dim)
        self.type_id_to_encoder[unspecified_type_id] = lookup_encoder
        self.entity_types[unspecified_mask] = unspecified_type_id
        self.entity_data_idx[unspecified_mask] = unspecified_index
        self.type_id_to_data[unspecified_type_id] = unspecified_index

        self.type_ids = torch.tensor(list(self.type_id_to_data.keys()),
                                     dtype=torch.long)

        for type_id, encoder in self.type_id_to_encoder.items():
            self.add_module(f'type_{type_id}_encoder', encoder)

        embeddings_buffer = torch.zeros([num_entities, dim], dtype=torch.float,
                                        requires_grad=False)
        self.register_buffer('embeddings_buffer', embeddings_buffer)

    def _plain_forward(self, indices: Optional[torch.LongTensor] = None,
                ) -> torch.Tensor:
        if self.training and indices is not None:
            batch_size = indices.shape[0]
            device = indices.device
            out = torch.empty([batch_size, self.shape[0]], dtype=torch.float,
                              device=device)

            # Sadly we have to move indices back to cpu to get property data
            indices = indices.cpu()
            entity_types = self.entity_types[indices]
            type_assignments = entity_types.unsqueeze(-1) == self.type_ids
            types_in_batch = torch.unique(entity_types).tolist()

            for t in types_in_batch:
                entity_type_mask = type_assignments[:, t]

                entities = indices[entity_type_mask]

                type_t_data = self.type_id_to_data[t]
                data = type_t_data[self.entity_data_idx[entities]]

                encoder = self.type_id_to_encoder[t]
                out[entity_type_mask] = encoder(data, device=device)

            # Update buffer
            self.embeddings_buffer[indices] = out.detach()
        else:
            # Use buffer during evaluation for speed
            if indices is None:
                return self.embeddings_buffer
            else:
                out = self.embeddings_buffer[indices]

        return out
