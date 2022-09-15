import torch
import json
import abc

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


def NoiseEmbedding(entities, random_seed: int = 42, dim=128, emb_range=(-1, 1)) -> LookupEmbedding:
    entity_to_id = build_entity_to_id(entities)

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

    return LookupEmbedding(embeddings=embeddings,
                           entity_to_id=entity_to_id,
                           metadata=metadata)


def MorganFingerpintEmbedding(mols, smiles) -> LookupEmbedding:
    print('fsdf')


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
