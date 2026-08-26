from pathlib import Path

from stuttermark.utils.config import load_config


def test_load_config(tmp_path: Path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("key: value\n")
    assert load_config(config_file) == {"key": "value"}
