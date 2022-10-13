from typing import Tuple, Mapping

from transformers import BertTokenizer
import torch
from torch import Tensor
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

        return (torch.tensor(entity_ids, dtype=torch.long),
                torch.tensor(rows, dtype=torch.long),
                torch.cat(all_tokens, dim=0))


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
        entities = data_dict['proteins']
        embeddings = data_dict['embeddings']

        entity_ids = []
        for e in entities:
            if e in entity_to_id:
                entity_ids.append(entity_to_id[e])

        entity_ids = torch.tensor(entity_ids, dtype=torch.long)
        rows = torch.arange(len(entity_ids))
        data = embeddings

        return entity_ids, rows, data

