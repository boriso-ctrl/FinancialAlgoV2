"""Financial Algorithms core package."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - packaging metadata
    __version__ = version("financial-algorithms")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
