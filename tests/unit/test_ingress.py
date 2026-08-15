"""HA ingress: X-Ingress-Path prefixes url_for while routing still matches.

Direct access (no header) must be byte-for-byte unchanged; behind HA ingress the
header must both keep routing working and make url_for emit the prefixed URL.
"""

from quart import Quart, url_for

from sargoshi.web import IngressMiddleware


def _app() -> Quart:
    app = Quart(__name__)

    @app.get("/ui/enrol")
    async def enrol() -> str:
        return url_for("enrol")  # what the templates emit

    return app


async def _get(asgi, path: str, headers: list) -> tuple[int, str]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "root_path": "",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
    }
    sent = [False]

    async def receive() -> dict:
        if not sent[0]:
            sent[0] = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    out: dict = {"status": None, "body": b""}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await asgi(scope, receive, send)
    return out["status"], out["body"].decode()


async def test_direct_access_is_unchanged():
    inner = _app()
    await inner.startup()
    try:
        status, body = await _get(IngressMiddleware(inner), "/ui/enrol", [])
        assert status == 200
        assert body == "/ui/enrol"  # no prefix without the header
    finally:
        await inner.shutdown()


async def test_ingress_header_prefixes_urls_and_still_routes():
    inner = _app()
    await inner.startup()
    try:
        status, body = await _get(
            IngressMiddleware(inner),
            "/ui/enrol",
            [(b"x-ingress-path", b"/api/hassio_ingress/abc123")],
        )
        assert status == 200  # route still matches under the prefix
        assert body == "/api/hassio_ingress/abc123/ui/enrol"  # url_for is prefixed
    finally:
        await inner.shutdown()


async def test_trailing_slash_in_header_is_normalised():
    inner = _app()
    await inner.startup()
    try:
        status, body = await _get(
            IngressMiddleware(inner),
            "/ui/enrol",
            [(b"x-ingress-path", b"/api/hassio_ingress/xyz/")],  # trailing slash
        )
        assert status == 200
        assert body == "/api/hassio_ingress/xyz/ui/enrol"  # no double slash
    finally:
        await inner.shutdown()
