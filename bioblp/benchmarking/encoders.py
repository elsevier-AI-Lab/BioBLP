import torch
import json
import pykeen
import pandas as pd
import numpy as np

from abc import ABC

from torch import Tensor

from pathlib import Path
from enum import Enum
from tqdm import tqdm

from typing import Tuple, List, Dict

from bioblp.models.bioblp import BioBLPTransE
from bioblp.models.bioblp import BioBLPComplEx
from bioblp.models.bioblp import BioBLPRotatE
from bioblp.logger import get_logger

logger = get_logger(__name__)


#
# Constants
#


NOISE = "noise"
STRUCTURAL = "structural"
COMPLEX = "complex"
TRANSE = "transe"
ROTATE = "rotate"
BIOBLPD = "bioblpd"
LABEL = "label"


class MissingValueMethod(Enum):
    DROP = "drop"
    MEAN = "mean"
    RANDOM = "random"


#
# Helpers
#


def get_random_tensor(shape, random_seed: int = 42, emb_range=(-1, 1)) -> Tensor:
    """ Generate random tensor, centered on mean of range.

        TODO: replace with random tensor to match distribution. 

    Parameters
    ----------
    shape : Tuple(int, int)
        Shape of tensor to generate.
    random_seed : int, optional
        Seed for RNG, by default 42
    emb_range : tuple, optional
        Range for distribution, by default (-1, 1)

    Returns
    -------
    Tensor
        Tensor containing randomly generated embeddings.
    """
    r1 = emb_range[0]
    r2 = emb_range[1]

    g = torch.Generator()
    g.manual_seed(random_seed)

    return (r1 - r2) * torch.rand(shape, generator=g) + r2


#
# Entity encoders to encode single entity type
#


class EntityEncoder(ABC):
    """ Abstract class for encoding entities with (pretrained) embeddings.

    Attributes
    ----------
    embeddings : Tensor
        Tensor holding entity embeddings of shape (n_entities, dim_emb).
    entity_to_id : Dict[str, int]
        Dictionary to map entity label, or external index, to internal index to fetch embedding.
    _emb_dim: int
        Internal helper to establish size of embedding.
    _emb_mean: Tensor
        Internal helper for mean imputation.
    _missing_value_cache : Dict[int, Tensor]
        Ensures consistency in imputed embedding assigned to recurring entities.
    """

    def __init__(self, embeddings: Tensor, entity_to_id: dict):
        """ Abstract class for encoding entities with (pretrained) embeddings.

        Parameters
        ----------
        embeddings : Tensor
            Tensor holding entity embeddings of shape (n_entities, dim_emb).
        entity_to_id : dict
            Dictionary to map entity label, or external index, to internal index to fetch embedding.
        """

        self.embeddings = embeddings
        self.entity_to_id = entity_to_id

        self._emb_dim = self.embeddings.size(1)
        self._emb_mean = torch.mean(self.embeddings, dim=0)
        self._missing_value_cache = {}

    def impute_missing_value(self, x, missing_value: MissingValueMethod) -> Tensor:
        """ Impute value for entity without pretrained embedding.

        Parameters
        ----------
        x : int
            Index for entity to replace with imputed value.
        missing_value : MissingValueMethod
            Method for imputation, options {MissingValueMethod.DROP, MissingValueMethod.RANDOM, MissingValueMethod.MEAN}.

        Returns
        -------
        Tensor
            Entity embedding with imputed values.
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
        """ Encode list of entities with pretrained embeddings, or with inputed value if embedding is missing.
            Return embeddings and positive mask indicating which entities were had embeddings.

        Parameters
        ----------
        x : List[str]
            List of entity external identifiers.
        missing_value : str, optional
            Method for imputation of missing embeddings, by default MissingValueMethod.MEAN.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - `embs` is the stacked entity embeddings
            - `mask` is the positive mask of entities having embeddings. Entities not in mask would have imputed embedding.
              This index reflects the external indices, ie index on the input `x`.
        """
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
    """ Implements random noise embeddings.

    Attributes
    ----------
    embeddings : Tensor
        Tensor holding entity embeddings of shape (n_entities, dim_emb).
    entity_to_id : Dict[str, int]
        Dictionary to map entity label, or external index, to internal index to fetch embedding.
    _emb_dim: int
        Internal helper to establish size of embedding.
    _emb_mean: Tensor
        Internal helper for mean imputation.
    _missing_value_cache : Dict[int, Tensor]
        Ensures consistency in imputed embedding assigned to recurring entities.
    """

    @classmethod
    def from_entities(cls, entities: List[str], random_seed: int = 42, dim: int = 128, emb_range: Tuple[int, int] = (-1, 1)):
        """ Initialise class from list of entities. Generate new set of embeddings from parameters.

        Parameters
        ----------
        entities : List[str]
            List of entities for which to generate embeddings.
        random_seed : int, optional
            Seed used in random number generator, by default 42.
        dim : int, optional
            Embedding size to use, by default 128.
        emb_range : Tuple[int, int], optional
            Range to use in RNG, by default (-1, 1).

        Returns
        -------
        NoiseEncoder
            Encoder for random noise
        """
        entity_set = set(entities)
        entity2id = {k: idx for idx, k in enumerate(sorted(list(entity_set)))}

        emb_shape = (len(entities), dim)

        embs = get_random_tensor(
            shape=emb_shape, random_seed=random_seed, emb_range=emb_range)

        return cls(embeddings=embs, entity_to_id=entity2id)


class ProteinEncoder(EntityEncoder):
    """ Implements encoding for proteins from pretrained ProtTrans model.

    Attributes
    ----------
    embeddings : Tensor
        Tensor holding entity embeddings of shape (n_entities, dim_emb).
    entity_to_id : Dict[str, int]
        Dictionary to map entity label, or external index, to internal index to fetch embedding.
    _emb_dim: int
        Internal helper to establish size of embedding.
    _emb_mean: Tensor
        Internal helper for mean imputation.
    _missing_value_cache : Dict[int, Tensor]
        Ensures consistency in imputed embedding assigned to recurring entities.
    """

    @classmethod
    def from_file(cls, emb_file: Path, entity_to_id: Path):
        """ Initialise class from pretrained protein ProtTrans model from pytorch file.

        Parameters
        ----------
        emb_file : Path
            Location of model file.
        entity_to_id : Path
            Identity mapping for entity label to ProtTrans index. This mapping is adopted in encoder.

        Returns
        -------
        ProteinEncoder
            Encode proteins with strucure transformer embedding.
        """
        logger.info(f"Loading protein data from {emb_file}...")

        embs = torch.load(emb_file)
        entity2id = {}

        with open(entity_to_id, "r") as f:
            entity2id = json.load(f)

        entity2id = {v: int(k) for k, v in entity2id.items()}
        return cls(embeddings=embs, entity_to_id=entity2id)


class MoleculeEncoder(EntityEncoder):
    """ Implements encoding for molecules from pretrained MolTrans model.

    Attributes
    ----------
    embeddings : Tensor
        Tensor holding entity embeddings of shape (n_entities, dim_emb).
    entity_to_id : Dict[str, int]
        Dictionary to map entity label, or external index, to internal index to fetch embedding.
    _emb_dim: int
        Internal helper to establish size of embedding.
    _emb_mean: Tensor
        Internal helper for mean imputation.
    _missing_value_cache : Dict[int, Tensor]
        Ensures consistency in imputed embedding assigned to recurring entities.
    """

    @classmethod
    def from_file(cls, emb_file: Path):
        """ Initialise class from pretrained molecule MolTrans model from pytorch file.

        Parameters
        ----------
        emb_file : Path
            Path to pytorch model file.

        Returns
        -------
        MoleculeEncoder
            Encode molecules with strucure transformer embedding.
        """
        logger.info(f"Loading molecule data from {emb_file}...")
        data_dict = torch.load(emb_file)

        emb_data = data_dict['embeddings']
        mol_ids = list(emb_data.keys())

        entity2row = {mol_i: idx for idx, mol_i in enumerate(mol_ids)}
        emb_records = [torch.from_numpy(
            np.mean(emb_data.get(x), axis=0)) for x in mol_ids]

        embs = torch.stack(emb_records, dim=0)

        return cls(embeddings=embs, entity_to_id=entity2row)


class KGEMEncoder(EntityEncoder):
    """ Implements encoding for entities with pretrained KGE model.

    Attributes
    ----------
    embeddings : Tensor
        Tensor holding entity embeddings of shape (n_entities, dim_emb).
    entity_to_id : Dict[str, int]
        Dictionary to map entity label, or external index, to internal index to fetch embedding.
    _emb_dim: int
        Internal helper to establish size of embedding.
    _emb_mean: Tensor
        Internal helper for mean imputation.
    _missing_value_cache : Dict[int, Tensor]
        Ensures consistency in imputed embedding assigned to recurring entities.
    """

    @classmethod
    def from_model(cls, model_file: Path, entity_to_id_file: Path, device: str = "cpu", filter_entities: List[str] = None):
        """ Initialise class from pretrained KGE model file.

        Parameters
        ----------
        model_file : Path
            Path to trained model file.
        entity_to_id_file : Path
            Path to entity id mapping for external model, entity label to index.
        device : str, optional
            Device to use for loading model, by default "cpu"
        filter_entities : List[str], optional
            Filter can be used to minimise memory footprint of Encoder by only keeping embeddings 
            for entities in filter, by default None

        Returns
        -------
        KGEMEncoder
            Encode entities with pretrained KG embedding.
        """
        logger.info(f"Loading model data from {model_file}...")

        model = torch.load(model_file, map_location=torch.device(device))
        entity2id_df = pd.read_csv(entity_to_id_file, sep="\t", header=0, compression="gzip")\
            .sort_values(by="id", ascending=True)

        entity_to_kgid = {k: v for (k, v) in entity2id_df[[
            "label", "id"]].values}

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

        if type(model) == pykeen.models.TransE:
            embs = model.entity_embeddings._embeddings(entity_idxs)
        elif type(model) in [BioBLPTransE, BioBLPComplEx, BioBLPRotatE]:
            embs = model.property_encoder.embeddings_buffer[entity_idxs]
        else:
            embs = model.entity_representations[0]._embeddings(entity_idxs)

        return cls(embeddings=embs, entity_to_id=entity_to_idx)


#
# Entity pair encoders
#


class EntityPairEncoder():
    """ Encoder for entity pairs.

    Attributes
    ----------
    left_encoder : EntityEncoder
        Encoder to use for first element (left) in pair.
    right_encoder : EntityEncoder
        Encoder for second element (right) in pair.
    """

    def __init__(self, left_encoder: EntityEncoder, right_encoder: EntityEncoder):
        """Encoder for entity pairs.

        Parameters
        ----------
        left_encoder : EntityEncoder
            Encoder to use for first element (left) in pair.
        right_encoder : EntityEncoder
            Encoder for second element (right) in pair.
        """

        self.left_encoder = left_encoder
        self.right_encoder = right_encoder

    def _combine(self, left_enc: Tensor, right_enc: Tensor, transform: str = "concat") -> Tensor:
        """ Internal method to combine element representations into a pair representation.
            Currently only supporting concatenation.

        Parameters
        ----------
        left_enc : Tensor
            Left entity embedding.
        right_enc : Tensor
            Right entity embedding.
        transform : str, optional
            Method for combination, by default "concat"

        Returns
        -------
        Tensor
            Combined entity representations.
        """
        if transform == "concat":
            return torch.cat([left_enc, right_enc], dim=1)

        # default
        return torch.cat([left_enc, right_enc], dim=1)

    def encode(self, x: np.array, missing_value: MissingValueMethod = MissingValueMethod.DROP, transform: str = "concat") -> Tensor:
        """ Encode entity pairs with respective encodings, impute missing embeddings and transform result.

        Parameters
        ----------
        x : np.array
            Input entity pairs in external labels.
        missing_value : MissingValueMethod, optional
            Approach for imputation of missing embeddings, by default MissingValueMethod.DROP.
        transform : str, optional
            Method to represent pair of embeddings, by default "concat".

        Returns
        -------
        Tensor
            Tensor for input entity pairs.
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


#
# Methods to assemble EntityPairEncoders
#


def RandomNoisePairEncoder(entities: List[str], random_seed: int) -> EntityPairEncoder:
    """ Build RandomNoisePairEncoder.

    Parameters
    ----------
    entities : List[str]
        List of entities for which to generate embeddings.
    random_seed : int
        Seed used in random number generator.

    Returns
    -------
    EntityPairEncoder
        Instance of EntityPairEncoder with RandomNoise embeddings
    """

    # prepare noisy embeddings
    noise_encoder = NoiseEncoder.from_entities(
        entities=entities, random_seed=random_seed)

    pair_encoder = EntityPairEncoder(
        left_encoder=noise_encoder, right_encoder=noise_encoder)
    return pair_encoder


def StructuralPairEncoder(proteins: Path, molecules: Path) -> EntityPairEncoder:
    """ Build encoder for structure data with ProtTrans embeddings for proteins
        and MolTrans embeddings for molecules.

    Parameters
    ----------
    proteins : Path
        Path to directory holding protein model.
    molecules : Path
        Path to directory holding molecul model.

    Returns
    -------
    EntityPairEncoder
        Instance of EntityPairEncoder with structural embeddings.
    """

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
    """ Build encoder for entities from KGE model.

    Parameters
    ----------
    model_dir : Path
        Path to directory with KGE model data.
    device : str, optional
        Device to load model, by default "cpu"
    entities : List[str], optional
        Filter can be used to minimise memory footprint of Encoder by only keeping embeddings 
        for entities in filter, by default None


    Returns
    -------
    EntityPairEncoder
        Instance of EntityPairEncoder with KG embeddings.
    """
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


def get_encoder(encoder_label: str, encoder_args: Dict[str, dict], entities: List[str]) -> EntityPairEncoder:
    """ Build encoder from label and config.

    Parameters
    ----------
    encoder_label : str
        Determines type of encoder to use, options: {'noise', 'structural', 'complex', 'transe', 'rotate'}
    encoder_args : Dict[str, dict]
        Arguments to pass to EntityPairEncoder.
    entities : List[str]
        Entity data.

    Returns
    -------
    EntityPairEncoder
        Instance of EntityPairEncoder defined by input encoder_label.
    """
    if encoder_label == NOISE:
        return RandomNoisePairEncoder(entities=entities, **encoder_args)
    elif encoder_label == STRUCTURAL:
        return StructuralPairEncoder(**encoder_args)
    elif encoder_label in {COMPLEX, TRANSE, ROTATE, BIOBLPD}:
        return KGEMPairEncoder(**encoder_args)
