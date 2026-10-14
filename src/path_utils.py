"""Keep machine-specific filesystem paths out of saved result summaries."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def result_path(path):
    """Return a project-relative path, or only the filename for external files."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name
