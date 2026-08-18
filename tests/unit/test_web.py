from sargoshi.web import create_app


async def test_health_is_ok(fake_pool, fake_speaker, config_service):
    client = create_app(fake_pool, fake_speaker, config_service).test_client()
    r = await client.get("/health")
    assert r.status_code == 200 and await r.get_json() == {"status": "ok"}


async def test_ready_true_when_warm(fake_pool, fake_speaker, config_service):
    client = create_app(fake_pool, fake_speaker, config_service).test_client()
    r = await client.get("/ready")
    body = await r.get_json()
    assert r.status_code == 200 and body["ready"] is True


async def test_ready_503_when_not_warm(fake_pool, fake_speaker, config_service):
    fake_pool._ready = False
    client = create_app(fake_pool, fake_speaker, config_service).test_client()
    r = await client.get("/ready")
    body = await r.get_json()
    assert r.status_code == 503 and body["ready"] is False


async def test_status_snapshot(fake_pool, fake_speaker, config_service):
    client = create_app(fake_pool, fake_speaker, config_service).test_client()
    r = await client.get("/status")
    body = await r.get_json()
    assert r.status_code == 200
    assert body["model"]["model"] == "large-v3-turbo"
    assert body["speaker"] == {"enabled": True, "ready": True, "model": "ecapa-tdnn"}
    assert body["frontends"]["wyoming"]["enabled"] is True
