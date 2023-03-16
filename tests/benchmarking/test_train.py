import torch
from bioblp.benchmarking.train import validate_features_exist
from bioblp.benchmarking.config import BenchmarkTrainConfig

from bioblp.logger import get_logger


logger = get_logger(__name__)


CONFIG_PATH = "conf/dpi-benchmark-train-20221216.toml"


def test_parse_train_config():
    cfg = BenchmarkTrainConfig.from_toml(CONFIG_PATH, run_id="abc")

    logger.info(cfg)


class TestValidateFeatures():

    models_conf = {
        "noise_lr": {
            "feature": "noise",
            "model": "LR"
        },
        "complex_lr": {
            "feature": "complex",
            "model": "LR"
        }
    }

    existing_feats = ["noise", "complex"]

    def setup_feats(self, dir):
        data = torch.arange(0., 12.).resize(3, 4)

        for feat in self.existing_feats:
            torch.save(data, dir.joinpath(f"{feat}.pt"))

    def test_validate_features_exist(self, tmp_path):
        dir = tmp_path.joinpath("features")
        dir.mkdir()
        self.setup_feats(dir)

        exists = validate_features_exist(dir, self.models_conf)

        assert exists is True

    def test_validate_features_exist_missing(self, tmp_path):
        dir = tmp_path.joinpath("features")
        dir.mkdir()
        self.setup_feats(dir)

        missing_feat = {
            "feature": "rotate",
            "model": "LR"
        }
        conf = self.models_conf
        conf.update({"rotate_LR": missing_feat})

        exists = validate_features_exist(dir, conf)

        assert exists is False
