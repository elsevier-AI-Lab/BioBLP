
import abc
import toml
import json

from dataclasses import dataclass, field
from typing import List
from pathlib import Path


def load_toml(toml_path: str) -> dict:
    toml_path = Path(toml_path)
    config = {}
    with open(toml_path, "r") as f:
        config = toml.load(f)

    return config


class ConfigJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # add conditional logic for any data structures that require special care
        # handling serialisation of Enum objects
        if isinstance(obj, Path):
            return str(obj.resolve())
        return json.JSONEncoder.default(self, obj)


@dataclass
class BenchmarkStepBaseConfig(abc.ABC):
    data_root: str
    experiment_root: str
    run_id: str
    outdir: str

    @classmethod
    def from_toml(cls, toml_path, run_id):
        raise NotImplementedError

    def resolve_outdir(self) -> Path:
        outdir = Path(self.data_root)\
            .joinpath(self.experiment_root)\
            .joinpath(self.run_id)\
            .joinpath(self.outdir)

        return outdir


@dataclass
class BenchmarkPreprocessConfig(BenchmarkStepBaseConfig):
    num_negs_per_pos: int
    kg_triples_dir: str

    @classmethod
    def from_toml(cls, toml_path: str, run_id: str):
        config_toml = load_toml(toml_path)

        cfg = config_toml.get("sampling")

        data_root = config_toml.get("data_root")
        experiment_root = config_toml.get("experiment_root")

        cfg.update({"data_root": data_root})
        cfg.update({"experiment_root": experiment_root})
        cfg.update({"run_id": run_id})

        return cls(**cfg)


@dataclass
class BenchmarkFeatureConfig(BenchmarkStepBaseConfig):
    transform: str
    missing_values: str
    encoders: list
    encoder_args: dict

    @classmethod
    def from_toml(cls, toml_path: str, run_id: str):
        conf_path = Path(toml_path)
        config_toml = load_toml(conf_path)

        data_root = config_toml.get("data_root")
        experiment_root = config_toml.get("experiment_root")

        cfg = config_toml.get("features")

        cfg.update({"data_root": data_root})
        cfg.update({"experiment_root": experiment_root})
        cfg.update({"run_id": run_id})

        return cls(**cfg)


@dataclass
class BenchmarkSplitConfig(BenchmarkStepBaseConfig):
    n_splits: int

    @classmethod
    def from_toml(cls, toml_path: str, run_id: str):
        conf_path = Path(toml_path)
        config_toml = load_toml(conf_path)

        data_root = config_toml.get("data_root")
        experiment_root = config_toml.get("experiment_root")

        cfg = config_toml.get("split")

        cfg.update({"data_root": data_root})
        cfg.update({"experiment_root": experiment_root})
        cfg.update({"run_id": run_id})

        return cls(**cfg)


@dataclass
class BenchmarkTrainConfig(BenchmarkStepBaseConfig):
    feature_dir: str
    splits_dir: str
    splits_file: str
    models: dict
    refit_params: List[str]
    n_iter: int = field(default=10, metadata={"help": "Number of HPO trials"})

    @classmethod
    def from_toml(cls, toml_path, run_id):
        conf = load_toml(toml_path=toml_path)
        cfg = {}

        cfg["models"] = conf.get("models")

        cfg.update(conf.get("train"))

        cfg["data_root"] = conf.get("data_root")
        cfg["experiment_root"] = conf.get("experiment_root")
        cfg["feature_dir"] = conf.get("features").get("outdir")
        cfg["splits_dir"] = conf.get("split").get("outdir")

        cfg.update({"run_id": run_id})

        return cls(**cfg)

    def resolve_feature_dir(self) -> Path:
        feature_dir = Path(self.data_root)\
            .joinpath(self.experiment_root)\
            .joinpath(self.run_id)\
            .joinpath(self.feature_dir)

        return feature_dir

    def resolve_splits_file(self) -> Path:
        splits_path = Path(self.data_root)\
            .joinpath(self.experiment_root)\
            .joinpath(self.run_id)\
            .joinpath(self.splits_dir)\
            .joinpath(self.splits_file)

        return splits_path
