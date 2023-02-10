from typing import Tuple, Mapping

from transformers import BertTokenizer
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np


class EntityPropertyPreprocessor:
    """Abstract class for preprocessing entity properties of different types
    into tensors suitable for machine learning wizardry."""
    def preprocess_file(self, file_path: str,
                        entity_to_id: Mapping[str, int]
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        """Read a file of entity properties, with one entity per line.
        Expects at each line an entity name, a tab, and a property to be
        encoded.

        Args:
            file_path: file mapping entities to properties
            entity_to_id: maps an entity name to an integer ID

        Returns:
            entity_ids: torch.Tensor containing entity IDs read by the method
            rows: torch.Tensor mapping each entity in entity_ids to a row in
                data
            data: torch.Tensor containing data for each entity in entity_ids
        """
        raise NotImplementedError


class TextEntityPropertyPreprocessor(EntityPropertyPreprocessor):
    """Preprocessor for entities with textual descriptions"""
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_file(self, file_path: str,
                        entity_to_id: Mapping[str, int]
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        all_tokens = []
        entity_ids = []
        rows = []
        row_count = 0
        with open(file_path) as file:
            for i, line in enumerate(tqdm(file, desc=f'Encoding {file_path}')):
                tab_idx = line.find('\t')
                entity, text = line[:tab_idx], line[tab_idx:].strip()

                if entity in entity_to_id:
                    tokens = self.tokenizer.encode(text,
                                                   max_length=self.max_length,
                                                   truncation=True,
                                                   padding='max_length',
                                                   return_tensors='pt')
                    all_tokens.append(tokens)
                    entity_id = entity_to_id[entity]
                    entity_ids.append(entity_id)
                    rows.append(row_count)
                    row_count += 1

        if len(all_tokens) > 0:
            all_tokens = torch.cat(all_tokens, dim=0)
        else:
            all_tokens = torch.tensor([], dtype=torch.long)

        return (torch.tensor(entity_ids, dtype=torch.long),
                torch.tensor(rows, dtype=torch.long),
                all_tokens)


class MolecularFingerprintPreprocessor(EntityPropertyPreprocessor):
    """Preprocessor for molecules with known molecular fingerprints"""
    def preprocess_file(self, file_path: str,
                        entity_to_id: Mapping[str, int]
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        all_fprints = []
        entity_ids = []
        rows = []
        row_count = 0
        with open(file_path) as file:
            for i, line in enumerate(tqdm(file, desc=f'Encoding {file_path}')):
                tab_idx = line.find('\t')
                entity, fprint = line[:tab_idx], line[tab_idx:].strip()

                if entity in entity_to_id:
                    fprint = torch.tensor(np.array(list(fprint), dtype=float), dtype=torch.float)
                    all_fprints.append(fprint)
                    entity_id = entity_to_id[entity]
                    entity_ids.append(entity_id)
                    rows.append(row_count)
                    row_count += 1

        return (torch.tensor(entity_ids, dtype=torch.long),
                torch.tensor(rows, dtype=torch.long),
                torch.stack(all_fprints, dim=0))


class PretrainedEmbeddingPreprocessor(EntityPropertyPreprocessor):
    def preprocess_file(self, file_path: str,
                        entity_to_id: Mapping[str, int]
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        data_dict = torch.load(file_path)
        entity_to_row = data_dict['identifiers']

        entity_ids = []
        data = []
        for entity, row in entity_to_row.items():
            if entity in entity_to_id:
                entity_ids.append(entity_to_id[entity])
                data.append(entity_to_row[entity])

        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        data_idx = torch.arange(len(entity_ids))
        data = torch.tensor(data, dtype=torch.long)

        return entity_ids, data_idx, data


class MoleculeEmbeddingPreprocessor(EntityPropertyPreprocessor):
    def preprocess_file(self, file_path: str,
                        entity_to_id: Mapping[str, int]
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        """Load embeddings for all the molecules we need, putting them
        in a single tensor that can be used to retrieve embeddings during
        training. Since molecules have variable length we use padding with
        a value of -1000 before placing them all inside a single 3D tensor
        of shape (N, L, D) where N is the number of molecules,
        L the maximum molecule length, and D the embedding dimension"""
        data_dict = torch.load(file_path)

        entity_ids = []
        data = []
        for molecule, embeddings in data_dict.items():
            if molecule in entity_to_id:
                entity_ids.append(entity_to_id[molecule])
                data.append(embeddings)

        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        data = pad_sequence(data, batch_first=True, padding_value=-10_000)
        data_idx = torch.arange(len(entity_ids))

        return entity_ids, data_idx, data
