import json
import toml

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
