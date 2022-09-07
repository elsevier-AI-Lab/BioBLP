from typing import List
import unittest
import tempfile
import os
import os.path as osp

import torch
from transformers import BertTokenizer

from bioblp.models.encoders import TransformerTextEncoder
import bioblp.loaders.preprocessors as preprocessors


class TestPropertyEncoders(unittest.TestCase):
    DISEASES = ['Irreversible FIBROSIS of the submucosal tissue of the MOUTH.',
                'The co-occurrence of pregnancy and parasitic diseases.',
                'Benign epidermal proliferations or tumors of viral in origin.',
                'Infections with bacteria of the genus PASTEURELLA.']

    MOLECULES = ['101010101010101010101010101010101010']

    def setUp(self):
        self.temp_file = None

    def tearDown(self):
        if self.temp_file is not None:
            if osp.exists(self.temp_file):
                os.remove(self.temp_file)

    def make_test_file(self, entities: List[int], choices: List[str]):
        if self.temp_file is None:
            file_name = tempfile.NamedTemporaryFile().name
            self.temp_file = file_name
        else:
            file_name = self.temp_file

        with open(file_name, 'w') as file:
            for i, entity in enumerate(entities):
                sample = choices[i % len(choices)]
                file.write(f'{entity}\t{sample}\n')

        return file_name

    def make_protein_test_file(self, emb_dim: int, entities: List[str]):
        if self.temp_file is None:
            file_name = tempfile.NamedTemporaryFile().name
            self.temp_file = file_name
        else:
            file_name = self.temp_file

        embeddings = torch.rand([len(entities), emb_dim])

        with open(file_name, 'w') as file:
            torch.save({'proteins': entities, 'embeddings': embeddings},
                       file_name)

        return file_name

    def test_text_preprocessor(self):
        entity_to_id = {str(i): i for i in range(10)}
        entities = list(entity_to_id.keys())
        file = self.make_test_file(entities, choices=self.DISEASES)

        max_length = 32
        tokenizer = BertTokenizer.from_pretrained(TransformerTextEncoder.BASE_MODEL)
        preprocessor = preprocessors.TextEntityPropertyPreprocessor(tokenizer,
                                                                    max_length)

        entities_tensor, data_idx, data = preprocessor.preprocess_file(file,
                                                                entity_to_id)
        self.assertEqual(len(entities_tensor), len(entities))
        self.assertEqual(len(data_idx), len(entities))
        self.assertTupleEqual(data.shape, (len(entities), max_length))

    def test_molecule_preprocessor(self):
        entity_to_id = {str(i): i for i in range(10)}
        entities = list(entity_to_id.keys())
        file = self.make_test_file(entities, choices=self.MOLECULES)

        preprocessor = preprocessors.MolecularFingerprintPreprocessor()
        entities_tensor, data_idx, data = preprocessor.preprocess_file(file,
                                                                       entity_to_id)

        self.assertEqual(len(entities_tensor), len(entities))
        self.assertEqual(len(data_idx), len(entities))
        self.assertTupleEqual(data.shape, (len(entities), len(self.MOLECULES[0])))

    def test_pretrained_protein_preprocessor(self):
        emb_dim = 32
        entity_to_id = {str(i): i for i in range(10)}
        entities = list(entity_to_id.keys())
        file = self.make_protein_test_file(emb_dim, entities)

        preprocessor = preprocessors.PretrainedEmbeddingPreprocessor()
        entities_tensor, data_idx, data = preprocessor.preprocess_file(file,
                                                                       entity_to_id)

        self.assertEqual(len(entities_tensor), len(entities))
        self.assertEqual(len(data_idx), len(entities))
        self.assertTupleEqual(data.shape, (len(entities), emb_dim))
