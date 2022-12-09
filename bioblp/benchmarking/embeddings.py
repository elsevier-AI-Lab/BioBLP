import torch
import json
import abc
import pandas as pd
import numpy as np

from argparse import ArgumentParser

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class LookupEmbedding():

    embeddings: torch.tensor
    entity_to_id: dict
    metadata: dict

    @classmethod
    def from_pretrained(cls, model_path):
        _embedding_file = "embeddings.pt"
        _e2id_file = "entity_to_id.tsv"
        _metadata_file = "metadata.json"

        model_path = Path(model_path)

        required = [_embedding_file, _e2id_file]

        for file_i in required:
            assert model_path.joinpath(file_i).exists(
            ), f"Missing required file {file_i} in dir {model_path}"

        embeddings = torch.load(model_path.joinpath(_embedding_file))
        entity_to_id = pd.read_csv(model_path.joinpath(
            _e2id_file), sep="\t", index_col=None, header=None)
        entity_to_id = {x[0]: x[1] for x in entity_to_id.values}

        metadata = None
        if model_path.joinpath(_metadata_file).exists():
            with open(model_path.joinpath(_metadata_file), "r") as f:
                metadata = json.load(f)

        return cls(embeddings=embeddings, entity_to_id=entity_to_id, metadata=metadata)

    def save(self, outdir: Union[str, Path], name: str):
        outdir = Path(outdir).joinpath(name)
        outdir.mkdir(parents=True, exist_ok=False)

        df = pd.DataFrame(
            [[k, v] for k, v in self.entity_to_id.items()], columns=["e_", "id_"])
        df.to_csv(outdir.joinpath("entity_to_id.tsv"),
                  sep="\t", index=False, header=False)

        torch.save(self.embeddings, outdir.joinpath("embeddings.pt"))

        if self.metadata is not None:
            with open(outdir.joinpath("metadata.json"), "w") as f:
                json.dump(self.metadata, f)


def save_as_pretrained_lookup_embedding(identifiers: list, tensors: torch.tensor, metadata: dict,
                                        outdir: Union[str, Path], outname: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir.joinpath(outname)

    meta = {}

    if metadata is not None or len(metadata) > 0:
        meta.update(metadata)

    data_dict = {
        "identifiers": identifiers,
        "embeddings": tensors,
        "metadata": meta
    }

    torch.save(data_dict, outfile)


def create_noise_lookup_emb(entities, random_seed: int = 42, dim: int = 128, emb_range=(-1, 1),
                            outdir=None, outname=None) -> dict:

    identifiers = sorted(list(set(entities)))
    # entity_to_id = {k: idx for idx, k in enumerate(sorted_unique_entities)}

    emb_shape = (len(identifiers), dim)

    r1 = emb_range[0]
    r2 = emb_range[1]

    g = torch.Generator()
    g.manual_seed(random_seed)

    embeddings = (r1 - r2) * torch.rand(emb_shape, generator=g) + r2

    metadata = {
        "random_seed": random_seed,
        "dim": dim,
        "emb_range": emb_range
    }

    # data_dict = {
    #     "identifiers" = identifiers,
    #     "embeddings" = embeddings,
    #     "metadata" = metadata
    # }

    if outdir is not None:
        assert outname is not None, "need outout name"

        save_as_pretrained_lookup_embedding(
            identifiers, embeddings, metadata, outdir, outname)

    return identifiers, embeddings, metadata


def create_noise_embeddings(entities, random_seed: int = 42, dim: int = 128, emb_range=(-1, 1),
                            outdir=None, outname=None) -> LookupEmbedding:

    entity_set = set(entities)
    entity_to_id = {k: idx for idx, k in enumerate(sorted(list(entity_set)))}

    emb_shape = (len(entities), dim)

    r1 = emb_range[0]
    r2 = emb_range[1]

    g = torch.Generator()
    g.manual_seed(random_seed)

    embeddings = (r1 - r2) * torch.rand(emb_shape, generator=g) + r2

    metadata = {
        "random_seed": random_seed,
        "dim": dim,
        "emb_range": emb_range
    }

    lookup = LookupEmbedding(embeddings=embeddings,
                             entity_to_id=entity_to_id,
                             metadata=metadata)

    if outdir is not None:
        assert outname is not None, "need outout name"

        # outdir = Path(outdir)

        # lookup.save(outdir, outname)

    return lookup


def create_moltrans_lookup_embedding(data_dir, outdir=None, outname=None) -> LookupEmbedding:
    """Create lookup embedding files from raw Molecular Transformer files.
    """
    drug_id_file = "drug_ids.txt"
    drug_smiles_file = "drug_smiles.txt"
    smiles_emb_file = "drug_smiles_full.npz"

    data_dir = Path(data_dir)

    drug_df = pd.read_csv(data_dir.joinpath(drug_id_file), header=None)
    drug_df.columns = ["db_id"]
    drug_smiles_df = pd.read_csv(
        data_dir.joinpath(drug_smiles_file), header=None)

    drug_df["smiles"] = drug_smiles_df

    drug_ids_sorted = sorted(drug_df['db_id'].values.tolist())
    id2smiles = {x[0]: x[1] for x in drug_df.values}

    # e2id
    entity_to_id = {idx: k for idx, k in enumerate(drug_ids_sorted)}

    emb_records = []
    with np.load(data_dir.joinpath(smiles_emb_file)) as data:
        for drug_id in drug_ids_sorted:
            smile = id2smiles.get(drug_id)

            emb_tensor = torch.tensor(data[smile])
            emb_mean_pool = torch.mean(emb_tensor, dim=0)  # is this best way?
            emb_records.append(emb_mean_pool)

    # embeddings
    embeddings = torch.stack(emb_records, dim=0)

    # meta
    metadata = {}

    lookup_embedding = LookupEmbedding(embeddings=embeddings,
                                       entity_to_id=entity_to_id,
                                       metadata=metadata)
    if outdir is not None:
        assert outname is not None, "need outout name"

        outdir = Path(outdir)

        lookup_embedding.save(outdir, outname)

    return lookup_embedding


def main(args):
    entities = Path(args.entities)
    outdir = Path(args.outdir)
    # name = args.name

    #
    # Load entities
    #

    df = pd.read_csv(entities, sep="\t", index_col=0)

    entities = np.concatenate((df["src"].values, df["tgt"].values))

    #
    # Noise
    #

    _, _, _ = create_noise_lookup_emb(entities=entities,
                                      outdir=outdir.joinpath(
                                          "noise_embeddings"),
                                      outname="noise_emb.pt")

    pass


if __name__ == "__main__":
    #
    # WIP
    #
    parser = ArgumentParser(
        description="Run lookup table generation procedure")
    parser.add_argument("--entities", type=str,
                        help="Path to pick up txt file of entities")
    parser.add_argument("--outdir", type=str, help="Path to write output")

    args = parser.parse_args()
    main(args)
