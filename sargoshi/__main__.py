"""sargoshi — speech-to-text and speaker-identification service."""

from __future__ import annotations

import argparse
import asyncio
import logging

from .config import Config, ConfigError, ConfigService, load_config
from .db import run_migrations
from .frontends.wyoming import run_wyoming
from .pool import ModelPool
from .speaker import SpeakerService
from .util import get_version
from .web import run_web

logger = logging.getLogger("sargoshi")


async def _boot(config_service: ConfigService) -> None:
    config = config_service.current
    pool = ModelPool(
        model_id=config.backend.model,
        device=config.backend.device,
        compute_type=config.backend.compute_type,
        download_root=config.storage.model_cache,
        cpu_threads=config.backend.cpu_threads,
        num_workers=config.backend.num_workers,
    )
    await pool.start()

    speaker: SpeakerService | None = None
    try:
        speaker = await _start_speaker(config)
        await _serve(config_service, pool, speaker)
    finally:
        if speaker is not None:
            await speaker.stop()
        await pool.stop()


async def _start_speaker(config: Config) -> SpeakerService | None:
    if not config.speaker.enabled:
        return None
    try:
        speaker = SpeakerService.from_config(config.speaker, model_cache=config.storage.model_cache)
        await asyncio.to_thread(run_migrations, config.speaker.db_path)
        await speaker.start()
        return speaker
    except Exception as e:
        logger.error("Speaker ID is enabled but failed to start: %s", e)
        raise SystemExit(1) from None


async def _serve(config_service: ConfigService, pool: ModelPool, speaker: SpeakerService | None) -> None:
    config = config_service.current
    tasks: list[asyncio.Task] = []
    if config.frontends.wyoming.enabled:
        tasks.append(asyncio.create_task(run_wyoming(config, pool, speaker), name="wyoming"))
    if config.frontends.ui.enabled:
        tasks.append(asyncio.create_task(run_web(config_service, pool, speaker), name="web"))

    if not tasks:
        logger.error("No frontends enabled; nothing to serve.")
        return

    logger.info("Frontends running: %s", ", ".join(t.get_name() for t in tasks))
    await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sargoshi",
        description="speech-to-text and speaker-identification service.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="YAML config file; seeded with defaults if missing (default: config.yaml)",
    )
    args = parser.parse_args()

    # Bootstrap at INFO so a config-load error is visible; the configured level
    # (logging.level in the YAML) is applied once the config is loaded.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
    except ConfigError as e:
        logger.error("Configuration error: %s", e)
        raise SystemExit(1) from None
    logging.getLogger().setLevel(config.logging.level)
    config_service = ConfigService(config, args.config)
    logger.info(
        "sargoshi %s starting — config=%s, model=%s, device=%s, compute=%s",
        get_version(),
        args.config,
        config.backend.model,
        config.backend.device,
        config.backend.compute_type,
    )
    try:
        asyncio.run(_boot(config_service))
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
