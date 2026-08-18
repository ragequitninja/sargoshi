import copy

import pytest

from sargoshi.config import Config, ConfigError
from sargoshi.speaker.embed import (
    ECAPAEmbedder,
    OpenVINOWavLMEmbedder,
    TorchWavLMEmbedder,
    create_embedder,
)

try:
    import transformers  # noqa: F401

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

try:
    from optimum.intel import OVModelForAudioXVector  # noqa: F401

    HAVE_OPTIMUM_INTEL = True
except ImportError:
    HAVE_OPTIMUM_INTEL = False


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


# Speaker devices are the union across models: ECAPA/TorchWavLM take the torch
# devices; the WavLM OpenVINO backend adds the openvino* targets. Config accepts the
# union — model x device compatibility is enforced when the embedder is built.
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda", "rocm", "auto", "openvino", "openvino:cpu", "openvino:gpu", "openvino:npu"],
)
def test_speaker_supported_devices(config, device):
    assert Config.from_dict(_with_device(config, "speaker", device)).speaker.device == device


@pytest.mark.parametrize("device", ["mps", "tpu", "openvino:tpu"])
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


# WavLM accepts the torch devices AND the openvino* targets; only genuinely unknown
# devices are rejected, and that check runs before the backend dependency, so it
# holds whether or not transformers/optimum-intel are installed.
@pytest.mark.parametrize("device", ["mps", "gpu", "banana", "openvino:tpu"])
def test_create_embedder_wavlm_rejects_unsupported_device(device):
    with pytest.raises(ValueError, match="does not support"):
        create_embedder("wavlm", device=device)


@pytest.mark.parametrize("alias", ["wavlm", "wavlm-base-plus-sv", "microsoft/wavlm-base-plus-sv"])
@pytest.mark.skipif(not HAVE_TRANSFORMERS, reason="needs the wavlm extra (transformers)")
def test_create_embedder_wavlm_builds_torch_from_alias(alias):
    emb = create_embedder(alias, device="cpu")
    assert isinstance(emb, TorchWavLMEmbedder)
    assert emb.model_id == "wavlm-base-plus-sv" and emb.dimension == 512


@pytest.mark.skipif(HAVE_TRANSFORMERS, reason="transformers installed; import guard cannot trigger")
def test_create_embedder_wavlm_without_transformers_is_loud():
    with pytest.raises(ImportError, match="transformers"):
        create_embedder("wavlm", device="cpu")


# The openvino* targets route to the OpenVINO backend, mapping to the OpenVINO
# runtime device string (openvino -> AUTO, openvino:gpu -> GPU, ...).
@pytest.mark.parametrize(
    ("device", "expected_ov"),
    [("openvino", "AUTO"), ("openvino:cpu", "CPU"), ("openvino:gpu", "GPU"), ("openvino:npu", "NPU")],
)
@pytest.mark.skipif(not HAVE_OPTIMUM_INTEL, reason="needs the wavlm-openvino extra (optimum-intel)")
def test_create_embedder_wavlm_openvino_routing(device, expected_ov):
    emb = create_embedder("wavlm", device=device)
    assert isinstance(emb, OpenVINOWavLMEmbedder)
    assert emb._ov_device == expected_ov
    assert emb.model_id == "wavlm-base-plus-sv" and emb.dimension == 512


@pytest.mark.skipif(HAVE_OPTIMUM_INTEL, reason="optimum-intel installed; import guard cannot trigger")
def test_create_embedder_wavlm_openvino_without_optimum_is_loud():
    with pytest.raises(ImportError, match="optimum-intel"):
        create_embedder("wavlm", device="openvino")


# ECAPA cannot use OpenVINO; the value parses at config load (union) but building the
# embedder rejects it loudly.
def test_create_embedder_ecapa_rejects_openvino():
    with pytest.raises(ValueError, match="does not support"):
        create_embedder("ecapa-tdnn", device="openvino")
