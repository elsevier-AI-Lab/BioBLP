import torch
import json

import pandas as pd
import numpy as np

from argparse import ArgumentParser
from dataclasses import asdict
from functools import reduce

from torch import Tensor

from pathlib import Path
from time import time
from tqdm import tqdm

from typing import Tuple, List, Dict

from bioblp.logger import get_logger
from bioblp.benchmarking.encoders import get_encoder
from bioblp.benchmarking.encoders import MissingValueMethod
from bioblp.benchmarking.encoders import EntityPairEncoder
from bioblp.benchmarking.encoders import EntityEncoder
from bioblp.benchmarking.encoders import NoiseEncoder
from bioblp.benchmarking.encoders import StructuralPairEncoder
from bioblp.benchmarking.encoders import RandomNoisePairEncoder
from bioblp.benchmarking.encoders import KGEMPairEncoder
from bioblp.benchmarking.config import BenchmarkFeatureConfig, ConfigJSONEncoder

from bioblp.data import COL_EDGE, COL_SOURCE, COL_TARGET
from bioblp.benchmarking.encoders import ROTATE, TRANSE, COMPLEX, STRUCTURAL, NOISE, LABEL


logger = get_logger(__name__)


#
# Building script
#


def save_features(outdir: Path, label: str, feature: Tensor, labels: Tensor):
    outfile = outdir.joinpath(f"{label}.pt")

    torch_obj = {"X": feature, "y": labels}
    torch.save(torch_obj, outfile)


def build_encodings(config: BenchmarkFeatureConfig, pairs: np.array, encoders: List[str],
                    encoder_args: Dict[str, dict], entities_filter: List[str]) -> Tuple[str, Tensor, Tensor]:
    encoded_bm = []

    for encoder_i_label in tqdm(encoders, desc=f"Encoding benchmarks..."):
        logger.info(f"Encoding with {encoder_i_label}")
        encoder_i_args = encoder_args.get(encoder_i_label)

        pair_encoder = get_encoder(encoder_i_label,
                                   encoder_i_args,
                                   entities=entities_filter)

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


def main(bm_file: str, conf: str, override_data_root=None, override_run_id=None):

    run_id = override_run_id or str(int(time()))

    config = BenchmarkFeatureConfig.from_toml(conf, run_id=run_id)

    if override_data_root is not None:
        config.data_root = override_data_root

    logger.info(
        f"Running process with config: {config} at time {run_id}...")

    # load benchmark data
    # here entities are strings

    bm_df = pd.read_csv(bm_file, sep='\t', names=[
                        COL_SOURCE, COL_EDGE, COL_TARGET, LABEL], header=0)

    pairs = bm_df[[COL_SOURCE, COL_TARGET]].values
    all_entities = np.unique(np.ravel(pairs)).tolist()

    labels = torch.from_numpy(bm_df[LABEL].values)

    # perform encodings
    encoded_bm = build_encodings(config=config,
                                 pairs=pairs,
                                 encoders=config.encoders,
                                 encoder_args=config.encoder_args,
                                 entities_filter=all_entities)

    # add plain benchmark data too
    encoded_bm.append(("raw", pairs, np.arange(len(pairs))))

    # common mask only when dropping missing embeddings
    if config.missing_values == MissingValueMethod.DROP.value:
        masked_encoded_bm, masked_labels = apply_common_mask(
            encoded_bm, labels)
    else:
        masked_encoded_bm = [(x[0], x[1]) for x in encoded_bm]
        masked_labels = labels

    feature_outdir = config.resolve_outdir()

    feature_outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving features to {feature_outdir}...")

    for enc_label, enc_pairs in masked_encoded_bm:
        logger.info(
            f"Saving {enc_label} features with shape: {enc_pairs.shape}")
        save_features(outdir=feature_outdir,
                      label=enc_label,
                      feature=enc_pairs,
                      labels=masked_labels)

    with open(feature_outdir.joinpath("config.json"), "w") as f:
        cfg_dict = asdict(config)
        json.dump(cfg_dict, f, cls=ConfigJSONEncoder)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate features for benchmark datasets")
    parser.add_argument("--conf", type=str,
                        help="Path to experiment configuration")
    parser.add_argument("--bm_file", type=str, help="Path to benchmark data")
    parser.add_argument("--override_data_root", type=str,
                        help="Path to root of data tree")
    parser.add_argument("--override_run_id", type=str,
                        help="Override run_id")

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    main(**vars(args))
