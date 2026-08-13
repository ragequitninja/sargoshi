import pytest

from sargoshi.config import ConfigError, load_config


async def test_set_persists_and_reloads(config_service, config_path):
    await config_service.set("backend.model", "small")
    assert config_service.current.backend.model == "small"
    assert "model: small" in config_path.read_text()
    # survives a fresh load from disk
    assert load_config(config_path).backend.model == "small"


async def test_update_multiple_keys_across_sections(config_service, config_path):
    await config_service.update({"speaker": {"threshold": 0.55}, "backend": {"beam_size": 1}})
    assert config_service.current.speaker.threshold == 0.55
    assert config_service.current.backend.beam_size == 1
    assert load_config(config_path).speaker.threshold == 0.55


async def test_invalid_change_rejected_leaves_memory_and_file_untouched(config_service, config_path):
    before = config_path.read_text()
    with pytest.raises(ConfigError):
        await config_service.set("backend.device", "tpu")
    assert config_service.current.backend.device != "tpu"
    assert config_path.read_text() == before, "file must be untouched on invalid change"


async def test_generic_over_any_key(config_service, config_path):
    await config_service.set("speaker.delivery", ["metadata", "inline"])
    assert config_service.current.speaker.delivery == ("metadata", "inline")
    assert load_config(config_path).speaker.delivery == ("metadata", "inline")


async def test_runtime_write_preserves_comments_and_formatting(config_service, config_path):
    await config_service.set("backend.model", "small")
    text = config_path.read_text()

    # value updated, old value gone
    assert "model: small" in text and "large-v3-turbo" not in text

    # comments survive: header, the changed line's own comment, and others
    for comment in (
        "# sargoshi test configuration",
        "# switch me via the UI",  # on the changed line
        "# my device choice",
        "# match threshold",
        "# inline flow map",
    ):
        assert comment in text, f"lost comment: {comment}"

    # untouched inline flow maps + list stay verbatim; null stays spelled "null"
    assert "{enabled: true, host: 0.0.0.0, port: 8080}" in text
    assert "delivery: [metadata, structured]" in text
    assert "language: null" in text
