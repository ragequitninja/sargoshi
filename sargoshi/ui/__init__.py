"""Management Web UI — a Quart blueprint on the shared app (Jinja2 + HTMX).

Current features:

    * switch the active model (from a list)
    * enrol speakers from uploaded WAV files
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from quart import Blueprint, Response, abort, render_template, request

from ..web import status_payload

if TYPE_CHECKING:
    from ..config import ConfigService
    from ..pool import ModelPool
    from ..speaker import SpeakerService

logger = logging.getLogger(__name__)

# Selectable multilingual Whisper models (the active one is always included too).
WHISPER_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]


def create_ui_blueprint(
    pool: ModelPool,
    speaker: SpeakerService | None = None,
    config_service: ConfigService | None = None,
) -> Blueprint:
    bp = Blueprint(
        "ui",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/ui/static",
    )

    def current_config():
        return config_service.current if config_service is not None else None

    def models_list() -> tuple[list[str], str]:
        active = pool.model_id
        return sorted(set(WHISPER_MODELS) | {active}), active

    async def speakers_list():
        if speaker is not None and speaker.enabled and speaker.ready:
            return await speaker.list_speakers()
        return []

    def active_speaker_model() -> str:
        """The embedder model new voiceprints are tagged with (for UI highlight)."""
        return speaker.model if (speaker is not None and speaker.enabled) else ""

    async def render_status(message: str = "", message_type: str = "ok"):
        return await render_template(
            "_status.html",
            status=status_payload(pool, speaker, current_config()),
            message=message,
            message_type=message_type,
        )

    async def render_speakers(message: str = "", message_type: str = "ok"):
        return await render_template(
            "_speakers.html",
            speakers=await speakers_list(),
            active_speaker_model=active_speaker_model(),
            message=message,
            message_type=message_type,
        )

    async def render_embeddings(speaker_id: str, message: str = "", message_type: str = "ok"):
        if speaker is None:
            return await render_speakers("Speaker ID is disabled.", "error")
        profile = await speaker.get_speaker(speaker_id)
        if profile is None:
            return await render_speakers("Speaker not found.", "error")
        return await render_template(
            "_embeddings.html",
            speaker=profile,
            embeddings=await speaker.list_embeddings(speaker_id),
            message=message,
            message_type=message_type,
        )

    # -- pages ---------------------------------------------------------

    @bp.get("/")
    async def index():
        models, active = models_list()
        return await render_template(
            "index.html",
            status=status_payload(pool, speaker, current_config()),
            models=models,
            active_model=active,
            speaker_enabled=(speaker is not None and speaker.enabled),
            speakers=await speakers_list(),
            active_speaker_model=active_speaker_model(),
        )

    @bp.get("/ui/status")
    async def status_partial():
        return await render_status()

    @bp.get("/ui/speakers")
    async def speakers_partial():
        return await render_speakers()

    # -- actions -------------------------------------------------------

    @bp.post("/ui/model")
    async def switch_model():
        form = await request.form
        model = (form.get("model") or "").strip()
        if not model:
            return await render_status("No model selected.", "error")
        try:
            await pool.switch_model(model)
        except Exception as e:
            logger.exception("Model switch to %r failed", model)
            return await render_status(f"Switch failed: {e}", "error")

        if config_service is not None:
            try:
                await config_service.set("backend.model", model)
            except Exception as e:
                logger.exception("Persisting model change failed")
                return await render_status(
                    f"Model switched to {pool.model_id}, but saving config failed: {e}",
                    "error",
                )
        return await render_status(f"Active model is now {pool.model_id} (saved).", "ok")

    @bp.post("/ui/enrol")
    async def enrol():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")

        form = await request.form
        files = await request.files
        name = (form.get("name") or "").strip()
        speaker_id = (form.get("speaker_id") or "").strip() or None
        attributes = {}
        for key in ("gender", "role", "language"):
            value = (form.get(key) or "").strip()
            if value:
                attributes[key] = value
        wavs = files.getlist("wav")

        # Adding to an existing speaker needs no name; a new profile does.
        if not speaker_id and not name:
            return await render_speakers("Name is required.", "error")
        if not wavs:
            return await render_speakers("Upload at least one WAV file.", "error")

        try:
            # Pass the ORIGINAL uploaded bytes — the store keeps them per embedding.
            audio = [await _read_upload(w) for w in wavs]
            result = await speaker.enrol(name=name, audio=audio, attributes=attributes, speaker_id=speaker_id)
        except Exception as e:
            logger.exception("Enrolment of %r failed", name or speaker_id)
            if speaker_id:
                return await render_embeddings(speaker_id, f"Failed: {e}", "error")
            return await render_speakers(f"Enrolment failed: {e}", "error")

        msg = f"Added {len(audio)} sample(s) to {result.name} ({result.embedding_count} total)."
        if speaker_id:
            return await render_embeddings(result.speaker_id, msg, "ok")
        return await render_speakers(msg, "ok")

    @bp.get("/ui/speakers/embeddings")
    async def manage_embeddings():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")
        sid = (request.args.get("id") or "").strip()
        if not sid:
            return await render_speakers("No speaker id given.", "error")
        return await render_embeddings(sid)

    @bp.post("/ui/embeddings/delete")
    async def delete_embedding():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")
        form = await request.form
        sid = (form.get("speaker_id") or "").strip()
        try:
            emb_id = int((form.get("id") or "").strip())
        except ValueError:
            return await render_embeddings(sid, "Invalid embedding id.", "error")
        try:
            affected = await speaker.delete_embedding(emb_id)
            if affected is None:
                return await render_embeddings(sid, "Voiceprint not found.", "error")
            return await render_embeddings(affected, "Voiceprint deleted.", "ok")
        except Exception as e:
            logger.exception("Delete embedding %s failed", emb_id)
            return await render_embeddings(sid, f"Delete failed: {e}", "error")

    @bp.get("/ui/embeddings/<int:embedding_id>/audio")
    async def embedding_audio(embedding_id: int):
        if speaker is None or not speaker.enabled:
            abort(404)
        data = await speaker.get_embedding_audio(embedding_id)
        if data is None:
            abort(404)
        return Response(
            data,
            mimetype="audio/wav",
            headers={"Content-Disposition": (f'attachment; filename="voiceprint-{embedding_id}.wav"')},
        )

    @bp.post("/ui/speakers/delete")
    async def delete_speaker():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")
        form = await request.form
        sid = (form.get("id") or "").strip()
        if not sid:
            return await render_speakers("No speaker id given.", "error")
        try:
            deleted = await speaker.delete_speaker(sid)
            if deleted:
                return await render_speakers("Speaker deleted.", "ok")
            return await render_speakers("Speaker not found.", "error")
        except Exception as e:
            logger.exception("Delete of %r failed", sid)
            return await render_speakers(f"Delete failed: {e}", "error")

    @bp.get("/ui/speakers/edit")
    async def edit_speaker():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")
        sid = (request.args.get("id") or "").strip()
        profile = await speaker.get_speaker(sid) if sid else None
        if profile is None:
            return await render_speakers("Speaker not found.", "error")
        return await render_template("_speaker_edit.html", s=profile)

    @bp.post("/ui/speakers/update")
    async def save_speaker():
        if speaker is None or not speaker.enabled:
            return await render_speakers("Speaker ID is disabled.", "error")
        form = await request.form
        sid = (form.get("id") or "").strip()
        name = (form.get("name") or "").strip()
        attributes = {}
        for key in ("gender", "role", "language"):
            value = (form.get(key) or "").strip()
            if value:
                attributes[key] = value
        if not sid:
            return await render_speakers("No speaker id given.", "error")
        if not name:
            return await render_speakers("Name is required.", "error")
        try:
            updated = await speaker.update_speaker(sid, name=name, attributes=attributes)
            if updated:
                return await render_speakers(f"Updated {name}.", "ok")
            return await render_speakers("Speaker not found.", "error")
        except Exception as e:
            logger.exception("Update of %r failed", sid)
            return await render_speakers(f"Update failed: {e}", "error")

    return bp


async def _read_upload(storage) -> bytes:
    """Read an uploaded file's bytes (Quart FileStorage.read may be sync or async)."""
    data = storage.read()
    if inspect.isawaitable(data):
        data = await data
    return data
