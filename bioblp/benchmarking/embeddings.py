import torch
import json
import abc
import pandas as pd

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


def create_noise_embeddings(entities, random_seed: int = 42, dim=128, emb_range=(-1, 1),
                            outdir=None, outname=None) -> LookupEmbedding:
    if not dim:
        dim = 128
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

        outdir = Path(outdir)

        lookup.save(outdir, outname)

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
    pass


if __name__ == "__main__":
    #
    # WIP
    #
    parser = ArgumentParser(description="Run model training procedure")
    parser.add_argument("--data_dir", type=str,
                        help="Path to pick up data")
    parser.add_argument("--outdir", type=str, help="Path to write output")
    parser.add_argument("--name", type=str, help="name of embeddings")

    args = parser.parse_args()
    main(args)
