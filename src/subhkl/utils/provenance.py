"""Run and build provenance for subhkl.

Every CLI step records how it was invoked -- the full command line, the
resolved values of all options (including the ones left at their defaults),
and the version/commit of subhkl that ran it. The record is printed when the
step starts and stored in the step's output file when it finishes, so a
result can always be traced back to the inputs and the build that produced
it. Records found in the input files of a step are carried over into its
output, so the last file of a chain holds the provenance of every step.

Inspect what a file carries with::

    python -m subhkl.io.parser provenance integrator.h5
"""

import functools
import getpass
import hashlib
import inspect
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time

from datetime import datetime
from datetime import timezone
from pathlib import Path

import h5py

# Name of the HDF5 group holding the provenance of all steps that contributed
# to a file, and of the sidecar written next to non-HDF5 outputs.
PROVENANCE_GROUP = "provenance"
SIDECAR_SUFFIX = ".provenance.json"

# Set by a packaging step (see the Dockerfile) when the build is made from a
# git checkout that is not shipped with the package.
COMMIT_ENV_VAR = "SUBHKL_GIT_COMMIT"
COMMIT_RESOURCE = Path(__file__).parent.parent / "resources" / "git_commit.txt"

HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx5")


def _run_git(*args: str) -> str | None:
    """Return the output of a git command run in the source tree, if possible."""
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@functools.lru_cache(maxsize=1)
def build_info() -> dict:
    """Return the version and source commit subhkl was built from.

    The commit is looked up, in order, in the ``SUBHKL_GIT_COMMIT``
    environment variable, in the file baked into the package at build time,
    and in the git checkout the package is imported from. It is ``"unknown"``
    when subhkl was installed from a source or wheel distribution that carries
    no commit information.
    """
    try:
        from subhkl._version import __version__ as version
    except ImportError:  # pragma: no cover - _version is generated at build
        version = "unknown"

    commit = os.environ.get(COMMIT_ENV_VAR, "").strip()
    source = "environment" if commit else ""

    if not commit and COMMIT_RESOURCE.is_file():
        commit = COMMIT_RESOURCE.read_text().strip()
        source = "package" if commit else ""

    dirty = None
    if not commit:
        commit = _run_git("rev-parse", "HEAD") or ""
        if commit:
            source = "git"
            status = _run_git("status", "--porcelain", "--untracked-files=no")
            dirty = bool(status) if status is not None else None

    info = {
        "subhkl_version": version,
        "git_commit": commit or "unknown",
        "git_commit_source": source or "unknown",
    }
    if dirty is not None:
        info["git_dirty"] = dirty
    return info


def _describe(value):
    """Render a parameter value as something JSON and HDF5 can both hold."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _describe(item) for key, item in value.items()}
    return repr(value)


def command_line() -> str:
    """Return the command line of this process, as it would be typed again.

    ``sys.argv[0]`` holds the path of the script being run, which is not what
    a user typed when subhkl is started as ``python -m subhkl.io.parser``, so
    that form is reconstructed here.
    """
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    name = getattr(spec, "name", None)
    if name:
        name = name.removesuffix(".__main__")
        return shlex.join([sys.executable, "-m", name, *sys.argv[1:]])
    return shlex.join(sys.argv)


def new_record(command: str, parameters: dict) -> dict:
    """Build the provenance record of a single step invocation."""
    record = {
        "command": command,
        "command_line": command_line(),
        "parameters": {name: _describe(value) for name, value in parameters.items()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "working_directory": os.getcwd(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    record.update(build_info())

    try:
        record["hostname"] = socket.gethostname()
    except OSError:
        pass
    try:
        record["user"] = getpass.getuser()
    except (OSError, KeyError):
        pass

    record["record_id"] = hashlib.sha1(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return record


def format_record(record: dict, indent: str = "") -> str:
    """Render a record as the block printed at the start of every step."""
    version = record.get("subhkl_version", "unknown")
    commit = record.get("git_commit", "unknown")
    if record.get("git_dirty"):
        commit += " (dirty)"

    lines = [
        f"subhkl {version} [{record.get('command', 'unknown')}]",
        f"  commit      : {commit}",
        f"  started     : {record.get('timestamp_utc', 'unknown')}",
        f"  host        : {record.get('hostname', 'unknown')}",
        f"  directory   : {record.get('working_directory', 'unknown')}",
        f"  command line: {record.get('command_line', 'unknown')}",
    ]
    if record.get("duration_seconds") is not None:
        lines.append(f"  duration    : {record['duration_seconds']:.1f} s")
    if record.get("status"):
        status = record["status"]
        if record.get("error"):
            status += f" ({record['error']})"
        lines.append(f"  status      : {status}")

    parameters = record.get("parameters") or {}
    if parameters:
        lines.append("  options:")
        width = max(len(name) for name in parameters)
        lines += [
            f"    {name:<{width}} = {json.dumps(value, default=str)}"
            for name, value in parameters.items()
        ]
    return "\n".join(indent + line for line in lines)


def print_record(record: dict) -> None:
    """Print a record, flushing so it precedes the output of the step itself."""
    print(format_record(record), flush=True)


def _write_group(parent: h5py.Group, name: str, record: dict) -> None:
    """Store one record as an HDF5 group of attributes."""
    group = parent.create_group(name)
    for key, value in record.items():
        if key == "parameters":
            continue
        group.attrs[key] = value if value is not None else "None"
    group.attrs["record_json"] = json.dumps(record, default=str)

    parameters = group.create_group("parameters")
    for key, value in (record.get("parameters") or {}).items():
        parameters.attrs[key] = (
            value if isinstance(value, (str, bool, int, float)) else json.dumps(value)
        )


def read_records(filename: str | os.PathLike) -> list[dict]:
    """Return the provenance records stored in a file, oldest step first.

    Works both for HDF5 outputs and for the JSON sidecar written next to
    outputs that are not HDF5 files. Returns an empty list for a file that
    carries no provenance.
    """
    path = Path(filename)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return []

    if not path.is_file():
        return []

    records = []
    try:
        with h5py.File(path, "r") as handle:
            group = handle.get(PROVENANCE_GROUP)
            if group is None:
                return []
            for name in sorted(group):
                raw = group[name].attrs.get("record_json")
                if raw is not None:
                    records.append(json.loads(raw))
    except Exception:  # noqa: BLE001 - a file we cannot read carries no records
        return []
    return records


def _inherited_records(inputs) -> list[dict]:
    """Collect the records of the steps that produced the inputs of a step."""
    inherited: list[dict] = []
    seen: set[str] = set()
    for filename in inputs:
        for record in read_records(filename):
            key = record.get("record_id") or json.dumps(record, sort_keys=True)
            if key not in seen:
                seen.add(key)
                inherited.append(record)
    return inherited


def write_records(filename: str | os.PathLike, records: list[dict]) -> None:
    """Store records in an output file, replacing any provenance it holds."""
    path = Path(filename)
    if not records or not path.is_file():
        return

    if path.suffix.lower() not in HDF5_SUFFIXES:
        sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
        sidecar.write_text(json.dumps(records, indent=2, default=str))
        return

    with h5py.File(path, "a") as handle:
        if PROVENANCE_GROUP in handle:
            del handle[PROVENANCE_GROUP]
        group = handle.create_group(PROVENANCE_GROUP)
        for index, record in enumerate(records):
            _write_group(group, f"{index:04d}_{record.get('command', 'step')}", record)


def stamp(filename, record: dict, inputs=()) -> None:
    """Store a step record, and the ones it inherits, in an output file."""
    try:
        write_records(filename, [*_inherited_records(inputs), record])
    except Exception as error:  # noqa: BLE001
        # Provenance must never cost a step the result it just computed.
        print(f"Warning: could not write provenance to {filename}: {error}")


def track(outputs=(), inputs=()):
    """Record the invocation of a CLI step and stamp it into its outputs.

    ``outputs`` and ``inputs`` name the parameters of the decorated command
    that hold its output and input file names. The record is printed before
    the step runs and written to every output file that exists afterwards,
    together with the records carried by the inputs. A step that raises is
    recorded too, with the error it failed on, since a file left behind by a
    failed run is exactly the one whose origin is hard to reconstruct.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)

            record = new_record(function.__name__.replace("_", "-"), parameters)
            print_record(record)

            started = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            else:
                record["status"] = "completed"
                return result
            finally:
                record["duration_seconds"] = round(time.monotonic() - started, 3)
                input_files = [
                    path
                    for name in inputs
                    for path in _as_paths(parameters.get(name))
                    if path
                ]
                for name in outputs:
                    for path in _as_paths(parameters.get(name)):
                        stamp(path, record, input_files)

        return wrapper

    return decorator


def _as_paths(value) -> list[str]:
    """Normalize a file name parameter to a list of paths."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]
