import copy

import pytest

from sargoshi.config import Config, ConfigError
from sargoshi.speaker.embed import ECAPAEmbedder, create_embedder


def _with_device(config, section, device):
    d = copy.deepcopy(config.to_dict())
    d[section]["device"] = device
    return d


# CTranslate2 STT backend: cpu/cuda/auto run; mps/rocm/openvino are not (yet).
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_backend_supported_devices(config, device):
    assert Config.from_dict(_with_device(config, "backend", device)).backend.device == device


@pytest.mark.parametrize("device", ["mps", "rocm", "openvino", "banana"])
def test_backend_rejects_unsupported_devices(config, device):
    with pytest.raises(ConfigError, match="backend.device"):
        Config.from_dict(_with_device(config, "backend", device))


# ECAPA speaker embedder: cpu/cuda/rocm/auto run; mps/openvino are not.
@pytest.mark.parametrize("device", ["cpu", "cuda", "rocm", "auto"])
def test_speaker_supported_devices(config, device):
    assert Config.from_dict(_with_device(config, "speaker", device)).speaker.device == device


@pytest.mark.parametrize("device", ["mps", "openvino", "tpu"])
def test_speaker_rejects_unsupported_devices(config, device):
    with pytest.raises(ConfigError, match="speaker.device"):
        Config.from_dict(_with_device(config, "speaker", device))


@pytest.mark.parametrize("section", ["backend", "speaker"])
def test_device_is_required(config, section):
    d = copy.deepcopy(config.to_dict())
    del d[section]["device"]
    with pytest.raises(ConfigError, match=f"{section}.device"):
        Config.from_dict(d)


@pytest.mark.parametrize("device", ["cpu", "cuda", "rocm", "auto"])
def test_create_embedder_accepts_supported(device):
    emb = create_embedder("ecapa-tdnn", device=device)
    assert isinstance(emb, ECAPAEmbedder) and emb._device == device


@pytest.mark.parametrize("device", ["mps", "openvino", "gpu", "banana"])
def test_create_embedder_rejects_unsupported_loudly(device):
    # same contract as the STT backend — no silent fallback, no generic "gpu"
    with pytest.raises(ValueError, match="does not support"):
        create_embedder("ecapa-tdnn", device=device)


def test_create_embedder_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown speaker model"):
        create_embedder("bogus-model", device="cpu")
