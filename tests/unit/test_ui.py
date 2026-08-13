"""Management UI blueprint (Quart + HTMX) end-to-end with fakes."""

import io

from quart.datastructures import FileStorage

from sargoshi.web import create_app


def _client(fake_pool, fake_speaker, config_service):
    return create_app(fake_pool, fake_speaker, config_service).test_client()


def _upload(audio, name="s.wav"):
    return FileStorage(stream=io.BytesIO(audio.make_wav()), filename=name, name="wav")


async def test_dashboard_renders(fake_pool, fake_speaker, config_service):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.get("/")
    html = await r.get_data(as_text=True)
    assert r.status_code == 200
    assert "sargoshi" in html
    assert '<select name="model"' in html
    assert "Enroll speaker" in html
    assert "large-v3-turbo" in html


async def test_status_partial(fake_pool, fake_speaker, config_service):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.get("/ui/status")
    assert r.status_code == 200 and "large-v3-turbo" in await r.get_data(as_text=True)


async def test_switch_model_persists_to_config(fake_pool, fake_speaker, config_service, config_path):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.post("/ui/model", form={"model": "small"})
    html = await r.get_data(as_text=True)
    assert r.status_code == 200 and "Active model is now small" in html
    assert fake_pool.model_id == "small"
    assert config_service.current.backend.model == "small"
    assert "model: small" in config_path.read_text()


async def test_enroll_then_speakers_table_shows_model_chip(fake_pool, fake_speaker, config_service, audio):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.post("/ui/enrol", form={"name": "John", "gender": "Male"}, files={"wav": _upload(audio)})
    html = await r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Added 1 sample(s) to John" in html and "gender=Male" in html

    r = await client.get("/ui/speakers")
    html = await r.get_data(as_text=True)
    assert "John" in html and "id01" in html and "Samples (1)" in html
    assert "ecapa-tdnn" in html  # the active model chip


async def test_manage_embeddings_add_download_delete(fake_pool, fake_speaker, config_service, audio):
    client = _client(fake_pool, fake_speaker, config_service)
    await client.post("/ui/enrol", form={"name": "John"}, files={"wav": _upload(audio)})

    r = await client.get("/ui/speakers/embeddings", query_string={"id": "id01"})
    html = await r.get_data(as_text=True)
    assert "John — voiceprints" in html and "download" in html and "Add more" in html

    # add a second sample to the existing speaker (stays in the manage view)
    r = await client.post("/ui/enrol", form={"speaker_id": "id01"}, files={"wav": _upload(audio, "s2.wav")})
    html = await r.get_data(as_text=True)
    assert "Added 1 sample(s) to John (2 total)" in html and "voiceprints" in html

    # download the stored WAV for embedding #1
    r = await client.get("/ui/embeddings/1/audio")
    body = await r.get_data()
    assert r.status_code == 200 and body[:4] == b"RIFF"
    assert "audio/wav" in r.headers.get("content-type", "")

    # delete embedding #1
    r = await client.post("/ui/embeddings/delete", form={"id": "1", "speaker_id": "id01"})
    assert "Voiceprint deleted" in await r.get_data(as_text=True)


async def test_delete_speaker(fake_pool, fake_speaker, config_service, audio):
    client = _client(fake_pool, fake_speaker, config_service)
    await client.post("/ui/enrol", form={"name": "John"}, files={"wav": _upload(audio)})
    r = await client.post("/ui/speakers/delete", form={"id": "id01"})
    html = await r.get_data(as_text=True)
    assert "Speaker deleted" in html and "No speakers enrolled yet" in html


async def test_edit_prefilled_then_update(fake_pool, fake_speaker, config_service, audio):
    client = _client(fake_pool, fake_speaker, config_service)
    await client.post("/ui/enrol", form={"name": "Jack", "gender": "Female"}, files={"wav": _upload(audio)})

    r = await client.get("/ui/speakers/edit", query_string={"id": "id01"})
    html = await r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'value="Jack"' in html and 'value="Female"' in html and "Save" in html

    r = await client.post(
        "/ui/speakers/update",
        form={"id": "id01", "name": "John", "gender": "Male", "language": "en"},
    )
    html = await r.get_data(as_text=True)
    assert "Updated John" in html
    assert "gender=Male" in html and "language=en" in html and "Female" not in html


async def test_edit_unknown_id(fake_pool, fake_speaker, config_service):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.get("/ui/speakers/edit", query_string={"id": "zzz"})
    assert "Speaker not found" in await r.get_data(as_text=True)


async def test_htmx_asset_served(fake_pool, fake_speaker, config_service):
    client = _client(fake_pool, fake_speaker, config_service)
    r = await client.get("/ui/static/htmx.min.js")
    assert r.status_code == 200 and int(r.headers.get("content-length", 0)) > 1000
