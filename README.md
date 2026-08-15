# sargoshi

Speech-to-text and speaker-identification service. One warm model pool behind multiple
protocol frontends and a management web UI to integrate via Home Assistant Apps.

### Protocol Frontends:

- **Wyoming for Home Assistant**
- **OpenAI-compatible REST** (Planned)
- **gRPC** (Planned)

### Speed-to-Text Backends:

- Ctranslate2 `aka` Faster-Whisper
- OpenVINO Intel (Planned)

### Speaker ID Backends:

- SpeechBrain
  - SpeechBrain is a low-cost speaker identification which can be 
    run on most CPUs without too much load. Accuracy on longer utterances
    is great with a reasonable number of embeddings (approximately 0.9% ERR)
    but suffers on shorter utterances.
  
- WavLM (Planned)
  - WavLM does much the same as SpeechBrain but with much better accuracy on shorter
    utterances. It may work on CPUs but requires a GPU to achieve any significant
    performance or reduced latency.

## Quickstart (Wyoming STT)

> **Python 3.11 or higher recommended.**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ctranslate2,speechbrain]"
```

**You must provide a `config.yaml`** — the service won't start without one (see config.example.yaml for a full
example with every key). Create one, then start the server:

```bash
python -m sargoshi                        # reads ./config.yaml
python -m sargoshi -c /data/config.yaml   # or point it elsewhere
```

Point Home Assistant's **Wyoming** integration at `host:10300` and it works as a drop-in
replacement for wyoming-faster-whisper.

