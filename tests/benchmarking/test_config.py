import pytest

from dataclasses import fields

from pathlib import Path
from bioblp.benchmarking.config import BenchmarkStepBaseConfig
from bioblp.benchmarking.config import BenchmarkPreprocessConfig
from bioblp.benchmarking.config import BenchmarkFeatureConfig
from bioblp.benchmarking.config import BenchmarkTrainConfig


from bioblp.logger import get_logger


logger = get_logger(__name__)

test_toml_file = Path(__file__).parent.joinpath("bm_test_conf.toml")


class TestBenchmarkStepBaseConfig():

    dr = "/home/skywalker/bioblp/data/"
    exp = "benchmark/experiments"
    step_out = "step_out"
    run_id = "123"

    def test_resolve_outdir(self):

        cfg = BenchmarkStepBaseConfig(
            data_root=self.dr,
            experiment_root=self.exp,
            run_id=self.run_id,
            outdir=self.step_out
        )

        full_outdir = cfg.resolve_outdir()

        assert str(full_outdir) == self.dr + self.exp + \
            "/" + self.run_id + "/" + self.step_out

    def test_test_resolve_outdir_mutated(self):
        cfg = BenchmarkStepBaseConfig(
            data_root=self.dr,
            experiment_root=self.exp,
            run_id=self.run_id,
            outdir=self.step_out
        )

        override_data_root = "/home/vader/bioblp/data/"

        cfg.data_root = override_data_root

        full_outdir = cfg.resolve_outdir()

        assert str(full_outdir) == override_data_root + self.exp + \
            "/" + self.run_id + "/" + self.step_out


class TestBenchmarkPreprocessConfig():

    def test_from_toml(self):
        expected_fields = ["data_root", "experiment_root", "run_id", "outdir",
                           "num_negs_per_pos", "kg_triples_dir"]

        run_id = "123"
        cfg = BenchmarkPreprocessConfig.from_toml(
            test_toml_file, run_id=run_id)

        cfg_fields = [field.name for field in fields(cfg)]

        assert cfg.num_negs_per_pos == 10
        assert cfg.data_root == "/home/skywalker/bioblp/"
        assert len(set(cfg_fields).difference(set(expected_fields))
                   ) == 0, f"Mismatch in fields: {set(cfg_fields).difference(set(expected_fields))}"

    def test_resolve_outdir(self):

        run_id = "123"
        cfg = BenchmarkPreprocessConfig.from_toml(
            test_toml_file, run_id=run_id)

        outdir = cfg.resolve_outdir()

        assert str(
            outdir) == f"/home/skywalker/bioblp/data/benchmarks/experiments/dpi_fda/20230224/{run_id}/sampled"


class TestBenchmarkFeatureConfig():

    def test_from_toml(self):
        expected_fields = ["data_root", "experiment_root", "run_id", "outdir",
                           "transform", "missing_values", "encoders", "encoder_args"]

        run_id = "123"
        cfg = BenchmarkFeatureConfig.from_toml(test_toml_file, run_id=run_id)

        cfg_fields = [field.name for field in fields(cfg)]

        assert len(set(cfg_fields).difference(set(expected_fields))
                   ) == 0, f"Mismatch in fields: {set(cfg_fields).difference(set(expected_fields))}"

    def test_resolve_outdir(self):

        run_id = "123"
        cfg = BenchmarkFeatureConfig.from_toml(test_toml_file, run_id=run_id)

        outdir = cfg.resolve_outdir()

        assert str(
            outdir) == f"/home/skywalker/bioblp/data/benchmarks/experiments/dpi_fda/20230224/{run_id}/features"


class TestBenchmarkTrainConfig():

    def test_from_toml(self):
        expected_fields = ["data_root", "experiment_root", "run_id", "outdir",
                           "feature_dir", "models", "refit_params", "n_iter", "splits_dir", "splits_file"]

        run_id = "123"
        cfg = BenchmarkTrainConfig.from_toml(test_toml_file, run_id=run_id)

        cfg_fields = [field.name for field in fields(cfg)]

        assert len(set(cfg_fields).difference(set(expected_fields))
                   ) == 0, f"Mismatch in fields: {set(cfg_fields).difference(set(expected_fields))}"

    def test_resolve_outdir(self):

        run_id = "123"
        cfg = BenchmarkTrainConfig.from_toml(test_toml_file, run_id=run_id)

        outdir = cfg.resolve_outdir()

        assert str(
            outdir) == f"/home/skywalker/bioblp/data/benchmarks/experiments/dpi_fda/20230224/{run_id}/models"

    def test_resolve_feature_outdir(self):

        run_id = "123"
        cfg = BenchmarkTrainConfig.from_toml(test_toml_file, run_id=run_id)

        outdir = cfg.resolve_feature_dir()

        assert str(
            outdir) == f"/home/skywalker/bioblp/data/benchmarks/experiments/dpi_fda/20230224/{run_id}/features"
