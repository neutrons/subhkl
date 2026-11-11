from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def get_version() -> str:
    """Return the dynamic subhkl package version generated via pyproject.toml / versioningit."""
    try:
        from subhkl._version import __version__ as _v  # type: ignore[import-not-found]

        if isinstance(_v, str) and _v.strip():
            return _v.strip()
    except Exception:
        pass
    try:
        v = _pkg_version("subhkl")
        if isinstance(v, str) and v.strip():
            return v.strip()
    except PackageNotFoundError:
        pass
    return "0.0.1"


__version__ = get_version()
