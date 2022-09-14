import torch
import json

from pathlib import Path
from typing import Union


class LookupEmbedding():
    def __init__(self, random_seed: int = 42, dim=128, emb_range=(-1, 1)):

        self.random_seed = random_seed

        self.embeddings = None
        self.dim = dim
        self.emb_range = emb_range
        self.entity_to_id = {}

    @classmethod
    def from_pretrained(cls, model_path):
        model_path = Path(model_path)
        cfg = {}
        with open(model_path.joinpath("config.json"), "r") as f:
            cfg = json.load(f)
            
        embeddings = torch.load(model_path.joinpath("embeddings.pt"))
        entity_to_id = pd.read_csv(model_path.joinpath("entity_to_id.tsv"), sep="\t", index_col = None, header=None)
        entity_to_id = {x[0]:x[1] for x in entity_to_id.values }
        
        obj = cls(random_seed = cfg["random_seed"],
                 dim=cfg["dim"],
                 emb_range=cfg["emb_range"])
        
        obj.embeddings = embeddings
        obj.entity_to_id = entity_to_id
        return obj

    def save(self, outdir: Union[str, Path], name="noise_embeddings"):
        outdir = Path(outdir).joinpath(name)
        outdir.mkdir(parents=True, exist_ok=False)
        
        config = {
            "random_seed": self.random_seed,
            "dim": self.dim,
            "emb_range": self.emb_range
        }
        
        with open(outdir.joinpath("config.json"), "w") as f:
            json.dump(config, f)
        
        df = pd.DataFrame([[k, v] for k, v in self.entity_to_id.items()], columns=["e_", "id_"])
        df.to_csv(outdir.joinpath("entity_to_id.tsv"), sep="\t", index=False, header=False)
        
        torch.save(self.embeddings, outdir.joinpath("embeddings.pt"))
        

    def _init_embeddings(self):
        raise NotImplementedError(
            "Child class needs implementation of _init_embeddings.")

    def _build_entity_to_id(self, entities):
        entities = sorted(dpi_entities)
        self.entity_to_id = {k: k_idx for k_idx, k in enumerate(entities)}

    def __call__(self, entities: list):
        if self.embeddings is None:
            self._build_entity_to_id(entities)
            self._init_embeddings()
        else:
            raise Error("Embedder already fit")


class NoiseEmbedding(LookupEmbedding):
    def __init__(self, random_seed: int = 42, dim=128, emb_range=(-1, 1)):
        super().__init__(random_seed=random_seed, dim=dim, emb_range=emb_range)

        self.dim = dim
        self.emb_range = emb_range
        self.entity_to_id = {}

    def _init_embeddings(self):
        emb_shape = (len(self.entity_to_id), self.dim)

        r1 = self.emb_range[0]
        r2 = self.emb_range[1]

        g = torch.Generator()
        g.manual_seed(self.random_seed)

        self.embeddings = (r1 - r2) * torch.rand(emb_shape, generator=g) + r2

    def _build_entity_to_id(self, entities):
        entities = sorted(entities)
        self.entity_to_id = {k: k_idx for k_idx, k in enumerate(entities)}

    def __call__(self, entities: list):
        self._build_entity_to_id(entities)
        self._init_embeddings()
