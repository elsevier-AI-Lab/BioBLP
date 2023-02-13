import torch
import json
import toml

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from functools import reduce

from torch import nn
from torch import Tensor

from pathlib import Path
from enum import Enum
from time import time
from tqdm import tqdm

from typing import Tuple, List, Dict

from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.logging import get_logger


logger = get_logger(__name__)


#
# Constants
#


NOISE = "noise"
STRUCTURAL = "structural"
COMPLEX = "complex"
TRANSE = "transe"
ROTATE = "rotate"
LABEL = "label"


class MissingValueMethod(Enum):
    DROP = "drop"
    MEAN = "mean"
    RANDOM = "random"


#
# Helpers
#


def get_random_tensor(shape, random_seed: int = 42, emb_range=(-1, 1)) -> Tensor:
    r1 = emb_range[0]
    r2 = emb_range[1]

    g = torch.Generator()
    g.manual_seed(random_seed)

    return (r1 - r2) * torch.rand(shape, generator=g) + r2


def load_toml(toml_path: str) -> dict:
    toml_path = Path(toml_path)
    config = {}
    with open(toml_path, "r") as f:
        config = toml.load(f)

    return config


class FeatureConfigJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # add conditional logic for any data structures that require special care
        # handling serialisation of Enum objects
        if isinstance(obj, Path):
            return str(obj.resolve())
        return json.JSONEncoder.default(self, obj)

#
# Entity encoders to encode single entity type
#


class EntityEncoder(ABC):
    """Written to resemble"""

    def __init__(self, embeddings: Tensor, entity_to_id: dict):

        self.embeddings = embeddings
        self.entity_to_id = entity_to_id

        self._emb_dim = self.embeddings.size(1)
        self._emb_mean = torch.mean(self.embeddings, dim=0)
        self._missing_value_cache = {}

    def impute_missing_value(self, x, missing_value: MissingValueMethod) -> Tensor:
        """can be {'drop', 'random', 'mean'}
        """
        emb = None

        if missing_value is MissingValueMethod.DROP:
            emb = torch.ones((self._emb_dim)) * float('-inf')

        elif missing_value is MissingValueMethod.RANDOM:
            if x in self._missing_value_cache:
                emb = self._missing_value_cache.get(x)
            else:
                # create new random
                # TODO: ensure distribution is similar to base embedding
                random_perturbation = get_random_tensor((self._emb_dim))
                emb = self._emb_mean + random_perturbation

                # store in cache
                self._missing_value_cache[x] = emb

        elif missing_value is MissingValueMethod.MEAN:
            emb = self._emb_mean

        return emb

    def encode(self, x: List[str], missing_value: str = MissingValueMethod.MEAN) -> Tuple[Tensor, Tensor]:
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
    def from_entities(cls, entities: List[str], random_seed: int = 42, dim: int = 128, emb_range: Tuple[int, int] = (-1, 1)):
        entity_set = set(entities)
        entity2id = {k: idx for idx, k in enumerate(sorted(list(entity_set)))}

        emb_shape = (len(entities), dim)

        embs = get_random_tensor(
            shape=emb_shape, random_seed=random_seed, emb_range=emb_range)

        return cls(embeddings=embs, entity_to_id=entity2id)


class ProteinEncoder(EntityEncoder):

    @classmethod
    def from_file(cls, emb_file: Path, entity_to_id: Path):
        logger.info(f"Loading protein data from {emb_file}...")

        embs = torch.load(emb_file)
        entity2id = {}

        with open(entity_to_id, "r") as f:
            entity2id = json.load(f)

        entity2id = {v: int(k) for k, v in entity2id.items()}
        return cls(embeddings=embs, entity_to_id=entity2id)


class MoleculeEncoder(EntityEncoder):

    @classmethod
    def from_file(cls, emb_file: Path):
        logger.info(f"Loading molecule data from {emb_file}...")
        data_dict = torch.load(emb_file)

        emb_data = data_dict['embeddings']
        mol_ids = list(emb_data.keys())

        entity2row = {mol_i: idx for idx, mol_i in enumerate(mol_ids)}
        emb_records = [torch.from_numpy(
            np.mean(emb_data.get(x), axis=0)) for x in mol_ids]

        embs = torch.stack(emb_records, dim=0)

        logger.info(embs.shape)

        return cls(embeddings=embs, entity_to_id=entity2row)


class KGEMEncoder(EntityEncoder):

    @classmethod
    def from_model(cls, model_file: Path, entity_to_id_file: Path, device: str = "cpu", filter_entities: List[str] = None):
        logger.info(f"Loading model data from {model_file}...")

        model = torch.load(model_file, map_location=torch.device(device))
        entity2id_df = pd.read_csv(entity_to_id_file, sep="\t", header=0, compression="gzip")\
            .sort_values(by="id", ascending=True)

        entity_to_kgid = {k: v for (k, v) in entity2id_df[[
            "label", "id"]].values}
        logger.info(len(entity_to_kgid))

        entity_kge_ids = []
        entity_to_idx = {}

        if filter_entities is not None:
            entity_kge_ids = []
            entity_to_idx = {}

            for ent_idx, ent_i in enumerate(sorted(filter_entities)):
                entity_kge_ids.append(entity_to_kgid.get(ent_i))
                entity_to_idx[ent_i] = ent_idx

            entity_idxs = torch.LongTensor(entity_kge_ids)

        else:
            entity_idxs = torch.LongTensor(entity2id_df["id"].values)
            entity_to_idx = entity_to_kgid

        embs = model.entity_representations[0]._embeddings(entity_idxs)

        logger.info(f"Loaded embeddings with shape: {embs.shape}")

        return cls(embeddings=embs, entity_to_id=entity_to_idx)


#
# Entity pair encoders
#


class EntityPairEncoder():
    def __init__(self, left_encoder: EntityEncoder, right_encoder: EntityEncoder):

        self.left_encoder = left_encoder
        self.right_encoder = right_encoder

    def _combine(self, left_enc: Tensor, right_enc: Tensor, transform: str = "concat"):
        if transform == "concat":
            return torch.cat([left_enc, right_enc], dim=1)
        # default
        return torch.cat([left_enc, right_enc], dim=1)

    def encode(self, x: np.array, missing_value: MissingValueMethod = MissingValueMethod.DROP, transform: str = "concat") -> Tensor:
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


def RandomNoisePairEncoder(entities: List[str], random_seed: int) -> EntityPairEncoder:

    # prepare noisy embeddings
    noise_encoder = NoiseEncoder.from_entities(
        entities=entities, random_seed=random_seed)

    pair_encoder = EntityPairEncoder(
        left_encoder=noise_encoder, right_encoder=noise_encoder)
    return pair_encoder


def StructuralPairEncoder(proteins: Path, molecules: Path) -> EntityPairEncoder:

    # prepare molecular embeddings
    mol_path = Path(molecules)
    mol_encoder = MoleculeEncoder.from_file(
        emb_file=mol_path.joinpath("biokg_molecule_embeddings.pt"))

    # prepare protein embeddings
    proteins_path = Path(proteins)
    prot_encoder = ProteinEncoder.from_file(
        emb_file=proteins_path.joinpath("protein_embeddings_full_24_12.pt"),
        entity_to_id=proteins_path.joinpath("biokg_uniprot_complete_ids.json"))

    pair_encoder = EntityPairEncoder(
        left_encoder=mol_encoder, right_encoder=prot_encoder)
    return pair_encoder


def KGEMPairEncoder(model_dir: Path, device: str = "cpu", entities: List[str] = None) -> EntityPairEncoder:
    model_dir = Path(model_dir)
    model_file = model_dir.joinpath("trained_model.pkl")

    entity_to_id_file = model_dir.joinpath(
        "training_triples/entity_to_id.tsv.gz")

    kgem_encoder = KGEMEncoder.from_model(model_file=model_file,
                                          entity_to_id_file=entity_to_id_file,
                                          device=device,
                                          filter_entities=entities)

    pair_encoder = EntityPairEncoder(
        left_encoder=kgem_encoder, right_encoder=kgem_encoder)

    return pair_encoder


def get_encoder(encoder_label: str, encoder_args: Dict[str, dict], entities: List[str]):
    if encoder_label == NOISE:
        return RandomNoisePairEncoder(entities=entities, **encoder_args)
    elif encoder_label == STRUCTURAL:
        return StructuralPairEncoder(**encoder_args)
    elif encoder_label in {COMPLEX, TRANSE, ROTATE}:
        return KGEMPairEncoder(**encoder_args)


#
# Building script
#


@dataclass
class FeatureConfig():
    data_root: str
    experiment_root: str
    outdir: str
    transform: str
    missing_values: str
    encoders: list
    encoder_args: dict


def parse_feature_conf(conf_path) -> dict:
    conf_path = Path(conf_path)
    config_toml = load_toml(conf_path)

    data_root = config_toml.get("data_root")
    experiment_root = config_toml.get("experiment_root")

    feat_config = config_toml.get("features")

    feat_config.update({"data_root": data_root})
    feat_config.update({"experiment_root": experiment_root})
    return feat_config


def save_features(outdir: Path, label: str, feature: Tensor, labels: Tensor):
    outfile = outdir.joinpath(f"{label}.pt")

    torch_obj = {"X": feature, "y": labels}
    torch.save(torch_obj, outfile)


def build_encodings(config: FeatureConfig, pairs: np.array, encoders: List[str], 
                    encoder_args: Dict[str, dict], entities_filter: List[str]) -> Tuple[str, Tensor, Tensor]:
    encoded_bm = []

    for encoder_i_label in tqdm(encoders, desc=f"Encoding benchmarks..."):
        logger.info(f"Encoding with {encoder_i_label}")
        encoder_i_args = encoder_args.get(encoder_i_label)
        pair_encoder = get_encoder(
            encoder_i_label, encoder_i_args, entities=entities_filter)

        missing_value_method = MissingValueMethod(config.missing_values)

        encoded_pairs, encoded_mask = pair_encoder.encode(pairs,
                                                          missing_value=missing_value_method,
                                                          transform=config.transform)

        encoded_bm.append((encoder_i_label, encoded_pairs, encoded_mask))
    return encoded_bm


def apply_common_mask(encoded_bm: List[Tuple[str, Tensor, Tensor]], labels: Tensor) -> Tuple[List[Tuple[str, Tensor]], Tensor]:
    logger.info("Masking features...")

    all_masks = [x[2] for x in encoded_bm]
    common_mask = torch.from_numpy(reduce(np.intersect1d, all_masks))

    logger.info(f"size after common mask {len(common_mask)}")

    masked_encoded_bm = []
    for enc_label, enc_pairs, _ in encoded_bm:
        masked_enc_pairs = enc_pairs[common_mask]
        masked_encoded_bm.append((enc_label, masked_enc_pairs))

    masked_labels = labels[common_mask]

    return masked_encoded_bm, masked_labels


def main(bm_file: str, conf: str, override_data_root=None, override_timestamp=None):

    config = parse_feature_conf(conf)

    if override_data_root is not None:
        config.update({"data_root": Path(override_data_root)})

    config = FeatureConfig(**config)

    timestamp = override_timestamp or str(int(time()))
    logger.info(
        f"Running process with config: {config} at time {timestamp}...")

    # load benchmark data
    # here entities are strings

    bm_df = pd.read_csv(bm_file, sep='\t', names=[
                        COL_SOURCE, COL_EDGE, COL_TARGET, LABEL], header=0)

    pairs = bm_df[[COL_SOURCE, COL_TARGET]].values
    all_entities = np.unique(np.ravel(pairs)).tolist()

    labels = torch.from_numpy(bm_df[LABEL].values)

    # perform encodings
    encoded_bm = build_encodings(config=config, pairs=pairs, encoders=config.encoders,
                                 encoder_args=config.encoder_args, entities_filter=all_entities)

    # add plain benchmark data too
    encoded_bm.append(("raw", pairs, np.arange(len(pairs))))

    # common mask only when dropping missing embeddings
    if config.missing_values == MissingValueMethod.DROP.value:
        masked_encoded_bm, masked_labels = apply_common_mask(encoded_bm, labels)
    else:
        masked_encoded_bm = [(x[0], x[1]) for x in encoded_bm]
        masked_labels = labels

    logger.info("Saving features...")

    feature_outdir = Path(config.data_root).joinpath(
        config.experiment_root).joinpath(timestamp)
    feature_outdir.mkdir(parents=True, exist_ok=True)

    for enc_label, enc_pairs in masked_encoded_bm:
        logger.info(
            f"Saving {enc_label} features with shape: {enc_pairs.shape}")
        save_features(outdir=feature_outdir,
                      label=enc_label,
                      feature=enc_pairs,
                      labels=masked_labels)

    with open(feature_outdir.joinpath("config.json"), "w") as f:
        cfg_dict = asdict(config)
        json.dump(cfg_dict, f, cls=FeatureConfigJSONEncoder)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate features for benchmark datasets")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--bm_file", type=str, help="Path to benchmark data")
    # parser.add_argument("--proteins", type=str, help="Path to protein data")
    # parser.add_argument("--molecules", type=str, help="Path to molecule data")
    # parser.add_argument("--complex", type=str, help="Path to complex data")
    # parser.add_argument("--transe", type=str, help="Path to transe data")
    # parser.add_argument("--rotate", type=str, help="Path to rotate data")
    parser.add_argument("--override_data_root", type=str,
                        help="Path to root of data tree")
    parser.add_argument("--override_timestamp", type=str,
                        help="Path to root of data tree")

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    main(**vars(args))
