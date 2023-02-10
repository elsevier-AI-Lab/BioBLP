import torch
import json

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from argparse import ArgumentParser

from torch import nn
from torch import Tensor

from pathlib import Path

from tqdm import tqdm

from typing import Tuple

from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.logging import get_logger

from bioblp.benchmarking.encoders import PretrainedLookupEncoder

logger = get_logger(__name__)


def get_random_tensor(shape, random_seed: int = 42, emb_range=(-1, 1)) -> Tensor:
    r1 = emb_range[0]
    r2 = emb_range[1]

    g = torch.Generator()
    g.manual_seed(random_seed)

    return (r1 - r2) * torch.rand(shape, generator=g) + r2


class EntityEncoder(ABC):
    """Written to resemble"""

    def __init__(self, embeddings: Tensor, entity_to_id: dict):

        self.embeddings = embeddings
        self.entity_to_id = entity_to_id

        self._emb_dim = self.embeddings.size(1)
        self._emb_mean = torch.mean(self.embeddings, dim=0)
        self._missing_value_cache = {}

    @classmethod
    def from_file(cls, emb_file, entity_to_id_file):
        raise NotImplementedError

    def impute_missing_value(self, x, missing_value: str) -> Tensor:
        """can be {'inf', 'random', 'avg'}
        """
        emb = None

        if missing_value == "inf":
            emb = torch.ones((self._emb_dim)) * float('-inf')

        elif missing_value == "random":
            if x in self._missing_value_cache:
                emb = self._missing_value_cache.get(x)
            else:
                # create new random
                emb = get_random_tensor((self._emb_dim))

                # store in cache
                self._missing_value_cache[x] = emb

        elif missing_value == "avg":
            emb = self._emb_mean

        return emb

    def encode(self, x: Tensor, missing_value: str = "avg") -> Tuple[Tensor, Tensor]:
        embs = []
        mask = []

        for idx, x_i in tqdm(enumerate(x), total=len(x)):
            if x_i in self.entity_to_id:
                # add to positive mask
                mask.append(idx)

                # collect embedding
                emb_i = self.embeddings[self.entity_to_id.get(x_i)]
                embs.append(emb_i)
            else:
                # return imputation strategy
                embs.append(self.impute_missing_value(
                    x_i, missing_value=missing_value))

        embs = torch.stack(embs)
        mask = torch.tensor(mask)
        return embs, mask


class NoiseEncoder(EntityEncoder):

    @classmethod
    def from_entities(cls, entities, random_seed: int = 42, dim=128, emb_range=(-1, 1)):
        entity_set = set(entities)
        entity2id = {k: idx for idx, k in enumerate(sorted(list(entity_set)))}

        emb_shape = (len(entities), dim)

        embs = get_random_tensor(
            shape=emb_shape, random_seed=random_seed, emb_range=emb_range)

        return cls(embeddings=embs, entity_to_id=entity2id)


class ProteinEncoder(EntityEncoder):

    @classmethod
    def from_file(cls, emb_file, entity_to_id):
        logger.info("loading protein data...")

        embs = torch.load(emb_file)
        entity2id = {}

        with open(entity_to_id, "r") as f:
            entity2id = json.load(f)

        entity2id = {v: int(k) for k, v in entity2id.items()}
        return cls(embeddings=embs, entity_to_id=entity2id)


class MoleculeEncoder(EntityEncoder):

    @classmethod
    def from_file(cls, emb_file):
        logger.info("loading molecule data...")
        data_dict = torch.load(emb_file)

        emb_data = data_dict['embeddings']
        mol_ids = list(emb_data.keys())

        entity2row = {mol_i: idx for idx, mol_i in enumerate(mol_ids)}
        emb_records = [torch.from_numpy(np.mean(emb_data.get(x), axis=0)) for x in mol_ids]

        embs = torch.stack(emb_records, dim=0)

        logger.info(embs.shape)
        
        return cls(embeddings=embs, entity_to_id=entity2row)


class EntityPairEncoder():
    def __init__(self, left_encoder: EntityEncoder, right_encoder: EntityEncoder):

        self.left_encoder = left_encoder
        self.right_encoder = right_encoder

    def _combine(self, left_enc, right_enc, transform="concat"):
        if transform == "concat":
            return torch.cat([left_enc, right_enc], dim=1)
        # default
        return torch.cat([left_enc, right_enc], dim=1)

    def encode(self, x, missing_value: str = "random", transform="concat") -> Tensor:
        """
        """
        left = x[:, 0]
        right = x[:, 1]

        left_enc, left_mask = self.left_encoder.encode(
            left, missing_value=missing_value)
        right_enc, right_mask = self.right_encoder.encode(
            right, missing_value=missing_value)

        total_enc = self._combine(left_enc, right_enc, transform)
        common_mask = torch.from_numpy(np.intersect1d(left_mask, right_mask))
        return total_enc, common_mask


def RandomNoisePairEncoder(entities) -> EntityPairEncoder:

    # prepare noisy embeddings
    noise_encoder = NoiseEncoder.from_entities(entities)

    pair_encoder = EntityPairEncoder(
        left_encoder=noise_encoder, right_encoder=noise_encoder)
    return pair_encoder


def StructuralPairEncoder(protein_path, mol_path) -> EntityPairEncoder:

    # prepare molecular embeddings
    mol_path = Path(mol_path)
    mol_encoder = MoleculeEncoder.from_file(emb_file=mol_path.joinpath("biokg_molecule_embeddings.pt"))

    # prepare protein embeddings
    proteins_path = Path(protein_path)
    prot_encoder = ProteinEncoder.from_file(emb_file=proteins_path.joinpath("protein_embeddings_full_24_12.pt"),
                                            entity_to_id=proteins_path.joinpath("biokg_uniprot_complete_ids.json"))

    pair_encoder = EntityPairEncoder(
        left_encoder=mol_encoder, right_encoder=prot_encoder)
    return pair_encoder


def build_features(bm_file: str, proteins: str, molecules: str, **kwargs):

    # load benchmark data
    # here entities are labels
    logger.info(bm_file, proteins)
    bm_df = pd.read_csv(bm_file, sep='\t', names=[
                        COL_SOURCE, COL_EDGE, COL_TARGET, "label"])
    logger.info(bm_df.head())

    src_entities = bm_df[COL_SOURCE].values
    tgt_entities = bm_df[COL_TARGET].values

    all_entities = list(src_entities) + list(tgt_entities)

    # noise test
    # noise_encoder = NoiseEncoder.from_entities(src_entities)
    # for each feature config
    # logger.info(noise_encoder)

    # encoded, masked = noise_encoder.encode(tgt_entities, missing_value="avg")
    # logger.info(encoded[:10])
    # logger.info(masked[:10])

    # proteins
    proteins_path = Path(proteins)

    # prot_encoder = ProteinEncoder.from_file(emb_file=proteins_path.joinpath("protein_embeddings_full_24_12.pt"),
    #                                         entity_to_id=proteins_path.joinpath("biokg_uniprot_complete_ids.json"))

    # # encoded, masked = prot_encoder.encode(tgt_entities, missing_value="avg")
    # # logger.info(tgt_entities[:10])
    # # logger.info(encoded[:10])
    # # logger.info(masked[:10])

    # # logger.info(prot_encoder.embeddings.shape)

       # ## mol_encoder

    # molecules = Path(molecules)

    # mol_encoder = MoleculeEncoder.from_file(emb_file=molecules.joinpath("biokg_molecule_embeddings.pt"))

    # mols_enc, mols_mask = mol_encoder.encode(src_entities)
    # logger.info(mols_enc[:3])
    # logger.info(mols_enc.shape)

    # ## pairs
    structural_encoder = StructuralPairEncoder(
        protein_path=proteins_path, mol_path=molecules)

    # pairs_encoder = EntityPairEncoder(left_encoder=noise_encoder, right_encoder=prot_encoder)

    pairs = bm_df[[COL_SOURCE, COL_TARGET]].head().values

    logger.info("===== struct encoder")

    pairs_enc, pairs_mask = structural_encoder.encode(
        pairs, transform="concat")

    logger.info(f"Mask length: {pairs_mask.shape}")
    logger.info(f"Mask length: {pairs_mask}")

    logger.info(f"enc length: {pairs_enc.shape}")
    logger.info(pairs_enc[:3])

    logger.info("===== noise encoder")
    noisy_encoder = RandomNoisePairEncoder(all_entities)

    pairs_enc, pairs_mask = noisy_encoder.encode(pairs)

    logger.info(f"Mask length: {pairs_mask.shape}")
    logger.info(f"Mask length: {pairs_mask}")

    logger.info(f"enc length: {pairs_enc.shape}")

 
    return


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run full benchmark experiment procedure")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--bm_file", type=str, help="Path to benchmark data")
    parser.add_argument("--proteins", type=str, help="Path to protein data")
    parser.add_argument("--molecules", type=str, help="Path to molecule data")
    parser.add_argument("--outdir", type=str, help="Path to write output")

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    build_features(**vars(args))
