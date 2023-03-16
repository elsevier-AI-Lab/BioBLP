import torch
import json
import pytest
import pandas as pd
import numpy as np

from bioblp.benchmarking.encoders import MissingValueMethod
from bioblp.benchmarking.encoders import EntityEncoder
from bioblp.benchmarking.encoders import NoiseEncoder
from bioblp.benchmarking.encoders import ProteinEncoder
from bioblp.benchmarking.encoders import MoleculeEncoder
from bioblp.benchmarking.encoders import EntityPairEncoder
from bioblp.benchmarking.encoders import KGEMEncoder
from bioblp.benchmarking.encoders import StructuralPairEncoder
from bioblp.benchmarking.encoders import RandomNoisePairEncoder
from bioblp.benchmarking.encoders import KGEMPairEncoder

from bioblp.benchmarking.encoders import get_random_tensor
from bioblp.benchmarking.encoders import get_encoder


from bioblp.logger import get_logger

logger = get_logger(__name__)


def test_get_random_tensor_single():

    single = get_random_tensor(shape=(1, 12), random_seed=1, emb_range=(-1, 1))

    assert len(single) == 1


def test_get_random_tensor_multi():

    multi = get_random_tensor(
        shape=(100, 12), random_seed=1, emb_range=(-1, 1))

    mean = torch.mean(multi, dim=0)

    assert abs(mean.sum().item()) < 0.1


class TestEntityEncoder():

    embs = torch.arange(0., 9.).resize(3, 3)

    entities = ["A", "B", "C"]
    entity_to_id = {"A": 0, "B": 1, "C": 2}

    def test_encode(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        encoded, _ = encoder.encode(self.entities)

        assert len(encoded) == len(self.entities)
        assert torch.sum((encoded - self.embs)) == 0

    def test_encode_repeated(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        entities_duplicates = self.entities + ["A"]
        encoded, _ = encoder.encode(x=entities_duplicates)

        assert len(encoded) == len(entities_duplicates)
        assert torch.sum((encoded[0] - encoded[-1])) == 0

    def test_encode_missing_value(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        entities_duplicates = self.entities + ["D"]
        encoded, _ = encoder.encode(x=entities_duplicates)

        assert len(encoded) == len(entities_duplicates)

    def test_encode_missing_value_with_cache(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        entities_duplicates = self.entities + ["D", "A", "D"]
        encoded, _ = encoder.encode(x=entities_duplicates)

        assert len(encoded) == len(entities_duplicates)
        assert torch.sum((encoded[-3] - encoded[-1])) == 0

    def test_impute_missing_value_drop(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        missing_entity = "D"
        missing = encoder.impute_missing_value(
            x=missing_entity, missing_value=MissingValueMethod.DROP)

        assert missing.size(0) == len(self.entities)
        assert torch.all(torch.isinf(missing))

    def test_impute_missing_value_random(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        missing_entity = "D"
        missing = encoder.impute_missing_value(
            x=missing_entity, missing_value=MissingValueMethod.RANDOM)

        assert missing.size(0) == len(self.entities)

    def test_impute_missing_value_mean(self):
        expected_mean = torch.mean(self.embs, dim=0)

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        missing_entity = "D"
        missing = encoder.impute_missing_value(
            x=missing_entity, missing_value=MissingValueMethod.MEAN)

        assert missing.size(0) == len(self.entities)
        assert torch.sum((missing - expected_mean)) == 0

    def test_encode_mask(self):

        encoder = EntityEncoder(embeddings=self.embs,
                                entity_to_id=self.entity_to_id)

        entities_duplicates = self.entities + ["D"]
        _, mask = encoder.encode(x=entities_duplicates)

        assert len(mask) == len(self.entities)
        assert all([a == b for a, b in zip(
            [entities_duplicates[x] for x in mask.tolist()], self.entities)])


class TestNoiseEncoder():

    entities = ["A", "B", "C"]
    dim = 6
    seed = 12
    emb_range = (-1, 1)

    def test_from_entities(self):

        encoder = NoiseEncoder.from_entities(entities=self.entities,
                                             random_seed=self.seed,
                                             dim=self.dim,
                                             emb_range=self.emb_range)

        encoded, _ = encoder.encode(x=self.entities)

        assert len(encoded) == len(self.entities)

    def test_from_entities_shape(self):

        encoder = NoiseEncoder.from_entities(entities=self.entities,
                                             random_seed=self.seed,
                                             dim=self.dim,
                                             emb_range=self.emb_range)

        expected_shape = (3, 6)

        assert encoder.embeddings.shape == expected_shape

    def test_from_entities_range(self):

        encoder = NoiseEncoder.from_entities(entities=self.entities,
                                             random_seed=self.seed,
                                             dim=self.dim,
                                             emb_range=(-10, 10))

        encoded, _ = encoder.encode(x=self.entities)

        assert torch.max(encoded) > 0


class TestProteinEncoder():
    embs = torch.arange(0., 9.).resize(3, 3)

    entities = ["A", "B", "C"]

    entity_to_id = {0: "A", 1: "B", 2: "C"}  # protein ids are stored inverted

    prot_file = "protein_embeddings.pt"
    e2id_file = "protein_ids.json"

    def _save_model(self, dir):
        torch.save(self.embs, dir.joinpath(self.prot_file))

        with open(dir.joinpath(self.e2id_file), "w") as f:
            json.dump(self.entity_to_id, f)

    def test_from_file(self, tmp_path):
        d = tmp_path.joinpath("proteins")
        d.mkdir()
        self._save_model(d)

        encoder = ProteinEncoder.from_file(emb_file=d.joinpath(self.prot_file),
                                           entity_to_id=d.joinpath(self.e2id_file))

        encoded, mask = encoder.encode(x=self.entities)

        assert len(mask) == len(self.entities)
        assert torch.sum((encoded - self.embs)) == 0


class TestMoleculeEncoder():
    embs = torch.arange(0., 12.).resize(3, 4)

    entities = ["A", "B", "C"]

    entity_to_id = {"A": 0, "B": 1, "C": 2}

    mol_file = "mol_embeddings.pt"

    def _save_model(self, dir):
        identifiers = self.entities
        emb_data = {k: self.embs.numpy() for k in self.entities}

        data = {
            "identifiers": identifiers,
            "embeddings": emb_data
        }

        torch.save(data, dir.joinpath(self.mol_file))

    def test_from_file(self, tmp_path):
        d = tmp_path.joinpath("mols")
        d.mkdir()
        self._save_model(d)

        encoder = MoleculeEncoder.from_file(emb_file=d.joinpath(self.mol_file))

        encoded, mask = encoder.encode(x=self.entities)

        assert len(mask) == len(self.entities)
        assert torch.sum((encoded - self.embs)) == 0


class TestKGEMEncoder():
    embs = torch.arange(0., 12.).resize(3, 4)

    entities = ["A", "B", "C"]

    entity_to_id = {"A": 0, "B": 1, "C": 2}

    prot_file = "kge_embeddings.pt"
    e2id_file = "kge_ids.tsv.gz"

    def _save_model(self, dir):
        torch.save(self.embs, dir.joinpath(self.prot_file))

        e2id_df = pd.DataFrame(
            [[k, v] for k, v in self.entity_to_id.items()], columns=["label", "id"])

        e2id_df.to_csv(dir.joinpath(self.e2id_file), sep="\t",
                       compression="gzip")

    @pytest.mark.skip(reason="Need to save dummy KGE model with pykeen.")
    def test_from_model(self, tmp_path):
        d = tmp_path.joinpath("kges")
        d.mkdir()
        self._save_model(d)

        encoder = KGEMEncoder.from_model(
            model_file=d.joinpath(self.prot_file),
            entity_to_id_file=d.joinpath(self.e2id_file),
            device="cpu",
            filter_entities=None)

        encoded, mask = encoder.encode(x=self.entities)

        assert len(mask) == len(self.entities)
        assert torch.sum((encoded - self.embs)) == 0


class TestEntityPairEncoder():

    pairs = np.array([["A", "C"], ["A", "B"], ["B", "C"]])
    entities = ["A", "B", "C"]

    def test_encode(self):
        single_dim = 3

        entity_encoder = NoiseEncoder.from_entities(
            self.entities, dim=single_dim)
        pair_encoder = EntityPairEncoder(
            left_encoder=entity_encoder, right_encoder=entity_encoder)

        encoded, _ = pair_encoder.encode(self.pairs)

        assert encoded.size(0) == len(self.entities)
        assert encoded.size(1) == 2 * single_dim

    def test_encode_masks(self):
        single_dim = 3

        left_entities = ["A", "B", "C"]
        right_entities = ["A", "C"]

        left_encoder = NoiseEncoder.from_entities(
            left_entities, dim=single_dim)

        right_encoder = NoiseEncoder.from_entities(
            right_entities, dim=single_dim)

        pair_encoder = EntityPairEncoder(
            left_encoder=left_encoder, right_encoder=right_encoder)

        _, mask = pair_encoder.encode(self.pairs)

        expected_mask = [0, 2]

        assert len(mask) == len(right_entities)
        assert all([a == b for a, b in zip(mask, expected_mask)])
