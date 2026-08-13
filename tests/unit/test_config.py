import copy

import pytest

from sargoshi.config import Config, ConfigError, load_config


def test_loads_complete_config(config):
    assert config.backend.model == "large-v3-turbo"
    assert config.frontends.ui.port == 8080
    assert config.speaker.enabled is True


def test_missing_file_errors_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError, match="not found"):
        load_config(missing)
    assert not missing.exists(), "load_config must never create a file"


def test_none_path_errors():
    with pytest.raises(ConfigError):
        load_config(None)


def test_missing_key_names_the_key(config):
    partial = copy.deepcopy(config.to_dict())
    del partial["backend"]["model"]
    with pytest.raises(ConfigError, match="backend.model"):
        Config.from_dict(partial)


def test_missing_section_names_the_section(config):
    missing = copy.deepcopy(config.to_dict())
    del missing["speaker"]
    with pytest.raises(ConfigError, match="speaker"):
        Config.from_dict(missing)


def test_empty_mapping_errors():
    with pytest.raises(ConfigError):
        Config.from_dict({})


def test_round_trips_through_dict(config):
    assert Config.from_dict(config.to_dict()) == config


def test_empty_file_errors(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_config(p)


def test_memory_db_path_rejected(config):
    bad = copy.deepcopy(config.to_dict())
    bad["speaker"]["db_path"] = ":memory:"
    with pytest.raises(ConfigError, match="in-memory"):
        Config.from_dict(bad)


def test_has_no_defaults_constructor():
    assert not hasattr(Config, "defaults")
