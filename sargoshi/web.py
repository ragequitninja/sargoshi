"""The Quart app hosting the HTTP enabled endpoints.

Routes:

    GET /         management - the management user interface
    GET /health   liveness - the process is up
    GET /ready    readiness - model loaded AND warm (drives the Docker HEALTHCHECK)
    GET /status   a richer operational snapshot (model, speaker, frontends)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from quart import Blueprint, Quart, jsonify

from .util import get_version

if TYPE_CHECKING:
    from .config import Config, ConfigService
    from .pool import ModelPool
    from .speaker import SpeakerService

logger = logging.getLogger(__name__)


def create_health_blueprint(
    pool: ModelPool,
    speaker: SpeakerService | None = None,
    config_service: ConfigService | None = None,
) -> Blueprint:
    bp = Blueprint("health", __name__)

    @bp.get("/health")
    async def health():
        return jsonify({"status": "ok"})

    @bp.get("/ready")
    async def ready():
        is_ready = pool.is_ready
        payload = {"ready": is_ready, "model": pool.model_id}
        return jsonify(payload), (200 if is_ready else 503)

    @bp.get("/status")
    async def status():
        config = config_service.current if config_service is not None else None
        return jsonify(status_payload(pool, speaker, config))

    return bp


def status_payload(pool: ModelPool, speaker: SpeakerService | None, config: Config | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "service": "sargoshi",
        "version": get_version(),
        "ready": pool.is_ready,
        "model": pool.status(),  # model, device, compute_type, loaded, ready
    }
    if speaker is not None:
        data["speaker"] = {"enabled": speaker.enabled, "ready": speaker.ready, "model": speaker.model}
    if config is not None:
        f = config.frontends
        data["frontends"] = {
            "wyoming": {"enabled": f.wyoming.enabled, "uri": f.wyoming.uri},
            "ui": {"enabled": f.ui.enabled, "host": f.ui.host, "port": f.ui.port},
        }
    return data


def create_app(
    pool: ModelPool,
    speaker: SpeakerService | None = None,
    config_service: ConfigService | None = None,
) -> Quart:
    from .ui import create_ui_blueprint

    app = Quart("sargoshi")
    app.register_blueprint(create_health_blueprint(pool, speaker, config_service))
    app.register_blueprint(create_ui_blueprint(pool, speaker, config_service))
    return app


class IngressMiddleware:
    """Mount the app under Home Assistant's ingress prefix.

    HA serves the add-on in an iframe under ``/api/hassio_ingress/<token>/`` and
    passes that prefix in the ``X-Ingress-Path`` header. We fold it into the ASGI
    scope's ``root_path`` and prefix the request path so routing still matches the
    real route while ``url_for`` emits ingress-prefixed URLs — which keeps the
    vendored htmx, every ``hx-*`` endpoint, and the upload POST inside the ingress
    session. Without the header (direct access, the Docker health check, Wyoming)
    it is a no-op, so direct access is unchanged.
    """

    def __init__(self, app: Quart) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            prefix = dict(scope.get("headers") or []).get(b"x-ingress-path", b"").decode("latin-1").rstrip("/")
            if prefix:
                scope = {
                    **scope,
                    "root_path": prefix,
                    "path": prefix + scope["path"],
                    "raw_path": prefix.encode("latin-1") + scope.get("raw_path", scope["path"].encode()),
                }
        await self._app(scope, receive, send)


async def run_web(
    config_service: ConfigService,
    pool: ModelPool,
    speaker: SpeakerService | None = None,
) -> None:
    app = create_app(pool, speaker, config_service)
    listener = config_service.current.frontends.ui
    hyper = HyperConfig()
    hyper.bind = [f"{listener.host}:{listener.port}"]
    # Don't let Hypercorn attach its own stderr handler (gunicorn-style format);
    # its `hypercorn.error` logger then just propagates to our root logger, so all
    # logs share one format instead of printing twice.
    hyper.errorlog = None
    logger.info(
        "Web app on http://%s:%d  (UI at /, health at /health /ready /status)",
        listener.host,
        listener.port,
    )
    # Never-resolving trigger so hypercorn doesn't install its own signal
    # handlers; shutdown happens via task cancellation from __main__.
    await serve(IngressMiddleware(app), hyper, shutdown_trigger=asyncio.Future)
