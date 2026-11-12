import contextlib
import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[2]
_SEMVER_RE = re.compile(r"^[vV]?\d+\.\d+")


def _git_exact_tag_or_branch() -> str | None:
    """Return the exact git tag at HEAD, or the current git branch name when inside a checkout."""
    env_ver = os.environ.get("SUBHKL_VERSION", "").strip()
    if env_ver:
        return env_ver
    if not (_REPO_DIR / ".git").exists():
        return None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        tag_res = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_DIR),
                "describe",
                "--tags",
                "--exact-match",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if tag_res.returncode == 0:
            tag = tag_res.stdout.strip()
            if tag:
                return tag.lstrip("vV") if _SEMVER_RE.match(tag) else tag
        branch_res = subprocess.run(
            ["git", "-C", str(_REPO_DIR), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if branch_res.returncode == 0:
            branch = branch_res.stdout.strip()
            if branch and branch != "HEAD":
                return branch
    return None


def get_version() -> str:
    """Return the exact git tag or active git branch name, falling back to package metadata."""
    git_ref = _git_exact_tag_or_branch()
    if git_ref:
        return git_ref
    with contextlib.suppress(ImportError, AttributeError, TypeError, ValueError):
        from subhkl._version import __version__ as _v  # type: ignore[import-not-found]

        if isinstance(_v, str) and _v.strip():
            return _v.strip()
    with contextlib.suppress(PackageNotFoundError, TypeError, ValueError):
        v = _pkg_version("subhkl")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "0.0.1"


def format_version_badge(version: str | None = None) -> str:
    """Format a version tag (`v1.5.0`) or git branch name (`feat/...`) for UI display."""
    raw = str(version if version is not None else get_version()).strip() or "0.0.1"
    if _SEMVER_RE.match(raw):
        return f"v{raw.lstrip('vV')}"
    return raw


def get_container_tag(version: str | None = None) -> str:
    """Return the Docker image tag matching .github/workflows/docker.yaml and publish.yaml."""
    env_tag = os.environ.get("SUBHKL_CONTAINER_TAG", "").strip()
    if env_tag:
        return env_tag
    raw = str(version if version is not None else get_version()).strip() or "latest"
    if _SEMVER_RE.match(raw):
        raw = raw.lstrip("vV")
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", raw).strip("-.")
    return sanitized or "latest"


def get_container_image(
    repo: str = "ghcr.io/neutrons/subhkl",
    version: str | None = None,
) -> str:
    """Return the full container image reference for the current git branch or version tag."""
    env_img = os.environ.get("SUBHKL_CONTAINER_IMAGE", "").strip()
    if env_img:
        return env_img
    return f"{repo}:{get_container_tag(version)}"


__version__ = get_version()
