from pathlib import Path

# Cache for version to avoid repeated file reading
_version_cache: str | None = None


def get_version() -> str:
    """
    Read the version from the version.txt file.

    This function reads the content safely without risk of code injection,
    as it only reads raw text and performs no evaluation.

    Returns:
        str:    The version from version.txt or 'unknown' if the file
                does not exist or cannot be read.
    """
    global _version_cache

    if _version_cache is not None:
        return _version_cache

    version_file = Path(__file__).parent.parent / "version.txt"

    try:
        file_version = version_file.read_text(encoding="utf-8").strip()
        _version_cache = file_version if file_version else "unknown"
    except (FileNotFoundError, PermissionError, OSError):
        _version_cache = "unknown"

    return _version_cache
