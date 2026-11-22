"""Three-tier shared-filesystem & heterogeneous mount-point resolution for SubHKL Galaxy runners.

Tier 1 (``SUBHKL_STAGING_DIR`` / ``TMPDIR``):
    Fast node-local NVMe scratch for step-local intermediates (e.g. per-run ``*.reduce.h5``).
Tier 2 (``SUBHKL_SHARED_DIR`` — mandatory in ``galaxy`` execution mode):
    Shared network filesystem (e.g. ``neutrondata.ornl.gov:/gpfs/neutronsfs/scratch``) mounted at
    potentially different local paths across runner hosts (``/scratch`` on ``imaginex-{1,2}`` vs.
    ``/gpfs/neutronsfs/scratch`` on ``analysis.sns.gov``). Large inter-step files (``merged.h5``,
    ephemeral ``peak_predictor.h5``) are stored here under unique filenames and exchanged between
    runners via tiny JSON pointer datasets in Galaxy History, then deleted after use.
Tier 3 (Galaxy Object Storage):
    Thin persisted deliverables (``solve.h5``, ``integrator.h5``, ``reflections.mtz``, Careless
    ``.mtz``) carrying ``source_files`` provenance so reduced data can be reconstructed on the fly.
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NFS mounts generally lack reliable POSIX byte-range locking (fcntl/lockd);
# disable HDF5 file locking by default so read-only runners never block or fail with ENOLCK.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SHARED_POINTER_MAGIC_KEY = "subhkl_shared_pointer"
SHARED_POINTER_VERSION = 1
_MAX_POINTER_FILE_BYTES = 256 * 1024
_STAGE_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_SHARED_UUID_FILENAME_RE = re.compile(r"^subhkl_shared_([0-9a-f]{32})(?:_|$)")

_NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "gpfs",
        "lustre",
        "cifs",
        "smb3",
        "ceph",
        "glusterfs",
        "panfs",
        "beegfs",
    }
)

_HOST_PREFIX_RE = re.compile(r"^([A-Za-z0-9._-]+):(/.*)$")
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{3})")


@dataclass(frozen=True)
class MountEntry:
    """Parsed entry from ``/proc/self/mountinfo``."""

    root: str
    mount_point: str
    fstype: str
    mount_source: str
    server: str | None
    export_path: str | None


def _unescape_mountinfo_token(token: str) -> str:
    return _OCTAL_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 8)), token)


def _normalize_ornl_export_path(server: str | None, export_path: str) -> str:
    """Canonicalize ORNL NFSv4 pseudo-root ``/scratch`` exports to ``/gpfs/neutronsfs/scratch``."""
    norm = posixpath.normpath(export_path)
    srv = (server or "").strip().lower()
    if srv.startswith("neutrondata") and (
        norm == "/scratch" or norm.startswith("/scratch/")
    ):
        suffix = norm[len("/scratch") :]
        return f"/gpfs/neutronsfs/scratch{suffix}"
    return norm


def _equivalent_export_paths(export_path: str) -> list[str]:
    """Return equivalent canonical and NFSv4-shorthand export paths for prefix matching."""
    norm = posixpath.normpath(export_path)
    out = [norm]
    gpfs_scratch = "/gpfs/neutronsfs/scratch"
    if norm == gpfs_scratch or norm.startswith(f"{gpfs_scratch}/"):
        alt = "/scratch" + norm[len(gpfs_scratch) :]
        if alt not in out:
            out.append(alt)
    elif norm == "/scratch" or norm.startswith("/scratch/"):
        alt = "/gpfs/neutronsfs" + norm
        if alt not in out:
            out.append(alt)
    return out


def _is_posix_prefix(path: str, prefix: str) -> bool:
    norm_p = posixpath.normpath(path)
    norm_pre = posixpath.normpath(prefix)
    if norm_pre == "/":
        return norm_p.startswith("/")
    return norm_p == norm_pre or norm_p.startswith(f"{norm_pre}/")


def _relative_posix_tail(path: str, prefix: str) -> str:
    norm_p = posixpath.normpath(path)
    norm_pre = posixpath.normpath(prefix)
    if norm_p == norm_pre:
        return ""
    if norm_pre == "/":
        return norm_p.lstrip("/")
    return norm_p[len(norm_pre) :].lstrip("/")


def parse_mountinfo(mountinfo_text: str | None = None) -> list[MountEntry]:
    """Parse ``/proc/self/mountinfo`` (or synthetic text) into ``MountEntry`` records."""
    if mountinfo_text is None:
        mountinfo_path = (
            os.environ.get("SUBHKL_MOUNTINFO_PATH") or "/proc/self/mountinfo"
        )
        if not os.path.isfile(mountinfo_path):
            return []
        try:
            with open(mountinfo_path, encoding="utf-8", errors="replace") as fh:
                mountinfo_text = fh.read()
        except OSError:
            return []

    entries: list[MountEntry] = []
    for raw_line in mountinfo_text.splitlines():
        line = raw_line.strip()
        if not line or " - " not in line:
            continue
        left_part, right_part = line.split(" - ", 1)
        left_fields = left_part.split()
        right_fields = right_part.split()
        if len(left_fields) < 5 or len(right_fields) < 2:
            continue
        root = _unescape_mountinfo_token(left_fields[3])
        mount_point = _unescape_mountinfo_token(left_fields[4])
        fstype = _unescape_mountinfo_token(right_fields[0]).lower()
        mount_source = _unescape_mountinfo_token(right_fields[1])

        server: str | None = None
        export_path: str | None = None
        m_srv = _HOST_PREFIX_RE.match(mount_source)
        if m_srv and not mount_source.startswith("/dev/"):
            server = m_srv.group(1)
            raw_export = m_srv.group(2)
            joined_export = posixpath.normpath(
                posixpath.join(raw_export, root.lstrip("/"))
            )
            export_path = _normalize_ornl_export_path(server, joined_export)
        elif (
            fstype == "gpfs"
            or mount_point == "/gpfs"
            or mount_point.startswith("/gpfs/")
        ):
            # Native IBM Spectrum Scale (GPFS) mount (e.g. /gpfs/neutronsfs on analysis.sns.gov
            # or a container bind mount from a GPFS filesystem where root != '/').
            if mount_point.startswith("/gpfs/"):
                export_path = posixpath.normpath(mount_point)
            else:
                fs_name = posixpath.basename(mount_source.rstrip("/"))
                if fs_name and fs_name not in ("", "/", "none", "gpfs"):
                    export_path = posixpath.normpath(
                        posixpath.join("/gpfs", fs_name, root.lstrip("/"))
                    )
        entries.append(
            MountEntry(
                root=root,
                mount_point=posixpath.normpath(mount_point),
                fstype=fstype,
                mount_source=mount_source,
                server=server,
                export_path=export_path,
            )
        )
    return entries


def split_network_path(path_or_netpath: str) -> tuple[str | None, str]:
    """Split ``[server:]/export/path`` into ``(server, '/export/path')``."""
    raw = str(path_or_netpath or "").strip()
    m_srv = _HOST_PREFIX_RE.match(raw)
    if m_srv:
        return m_srv.group(1), posixpath.normpath(m_srv.group(2))
    return None, posixpath.normpath(raw) if raw.startswith("/") else os.path.abspath(
        raw
    )


def to_shared_network_path(
    local_or_net_path: str | os.PathLike[str],
    *,
    mountinfo_text: str | None = None,
) -> str:
    """Convert a local shared-filesystem path into its canonical full network path.

    Examples:
    - On ``imaginex-1`` where ``/scratch`` mounts ``neutrondata.ornl.gov:/gpfs/neutronsfs/scratch``:
      ``/scratch/subhkl/abc.h5`` -> ``neutrondata.ornl.gov:/gpfs/neutronsfs/scratch/subhkl/abc.h5``
    - On ``analysis.sns.gov`` where ``/gpfs/neutronsfs`` is native GPFS:
      ``/gpfs/neutronsfs/scratch/subhkl/abc.h5`` -> ``/gpfs/neutronsfs/scratch/subhkl/abc.h5``
    """
    raw_str = os.fspath(local_or_net_path).strip()
    srv, norm_path = split_network_path(raw_str)
    if srv is not None:
        canon_exp = _normalize_ornl_export_path(srv, norm_path)
        return f"{srv}:{canon_exp}"

    lookup_path = (
        posixpath.normpath(raw_str)
        if mountinfo_text is not None
        else posixpath.normpath(os.path.realpath(raw_str))
    )
    entries = parse_mountinfo(mountinfo_text)
    best_entry: MountEntry | None = None
    for entry in entries:
        if entry.export_path is None:
            continue
        if _is_posix_prefix(lookup_path, entry.mount_point) and (
            best_entry is None or len(entry.mount_point) > len(best_entry.mount_point)
        ):
            best_entry = entry

    if best_entry is not None and best_entry.export_path is not None:
        tail = _relative_posix_tail(lookup_path, best_entry.mount_point)
        full_export = (
            posixpath.normpath(posixpath.join(best_entry.export_path, tail))
            if tail
            else best_entry.export_path
        )
        full_export = _normalize_ornl_export_path(best_entry.server, full_export)
        if best_entry.server:
            return f"{best_entry.server}:{full_export}"
        return full_export

    return lookup_path


def _servers_compatible(net_server: str | None, mount_server: str | None) -> bool:
    if not net_server or not mount_server:
        return True
    s1 = net_server.strip().lower()
    s2 = mount_server.strip().lower()
    if s1 == s2:
        return True
    return s1.split(".", 1)[0] == s2.split(".", 1)[0]


def resolve_shared_local_path(
    path_or_netpath: str | os.PathLike[str],
    *,
    mountinfo_text: str | None = None,
    shared_dir_env: str | None = None,
    must_exist: bool = False,
) -> str:
    """Resolve a full network path (or local path) to the local mount point on this host."""
    raw_str = os.fspath(path_or_netpath).strip()
    if not raw_str:
        raise ValueError("Shared path cannot be empty.")

    net_server, remote_path = split_network_path(raw_str)
    canon_remote = _normalize_ornl_export_path(net_server, remote_path)
    cand_exports = _equivalent_export_paths(canon_remote)
    if remote_path not in cand_exports:
        cand_exports.append(remote_path)

    entries = parse_mountinfo(mountinfo_text)
    use_real_fs = mountinfo_text is None

    # 1. Match against /proc/self/mountinfo network/GPFS export roots (longest export match wins).
    scored_matches: list[tuple[int, int, str]] = []
    for entry in entries:
        if entry.export_path is None:
            continue
        entry_exports = _equivalent_export_paths(entry.export_path)
        for cand_exp in cand_exports:
            for m_exp in entry_exports:
                if _is_posix_prefix(cand_exp, m_exp):
                    tail = _relative_posix_tail(cand_exp, m_exp)
                    local_cand = (
                        posixpath.normpath(posixpath.join(entry.mount_point, tail))
                        if tail
                        else entry.mount_point
                    )
                    srv_score = (
                        2
                        if (
                            net_server
                            and entry.server
                            and _servers_compatible(net_server, entry.server)
                        )
                        else (1 if _servers_compatible(net_server, entry.server) else 0)
                    )
                    scored_matches.append((srv_score, len(m_exp), local_cand))

    missing_mount_cand: str | None = None
    if scored_matches:
        scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_local = scored_matches[0][2]
        if not use_real_fs:
            return best_local
        for _, _, cand_local in scored_matches:
            if os.path.exists(cand_local):
                return cand_local
        for _, _, cand_local in scored_matches:
            _revalidate_nfs_parent_dir(os.path.dirname(cand_local))
            if os.path.exists(cand_local):
                return cand_local
        if not must_exist:
            return best_local
        missing_mount_cand = best_local

    # 2. Check non-root mount points in mountinfo matching canon_remote directly
    # (e.g. native GPFS or bind mount at /gpfs/neutronsfs without NFS server: source).
    nonroot_mounts = [
        e.mount_point
        for e in entries
        if e.mount_point != "/"
        and (
            e.fstype in _NETWORK_FSTYPES
            or e.mount_point.startswith(("/gpfs", "/scratch", "/SNS", "/HFIR", "/mnt"))
        )
    ]
    for cand_exp in cand_exports:
        for mp in sorted(nonroot_mounts, key=len, reverse=True):
            if _is_posix_prefix(cand_exp, mp):
                if not use_real_fs or os.path.exists(cand_exp) or not must_exist:
                    return cand_exp
                if missing_mount_cand is None:
                    missing_mount_cand = cand_exp

    # 3. Direct local filesystem existence check (when running against real host FS).
    if use_real_fs:
        for cand_exp in ([remote_path] if net_server is None else []) + cand_exports:
            if os.path.exists(cand_exp) or os.path.isdir(os.path.dirname(cand_exp)):
                if must_exist and not os.path.exists(cand_exp):
                    continue
                return os.path.abspath(cand_exp)

    # 4. Fallback via runner's configured SUBHKL_SHARED_DIR (supports local/synthetic non-NFS dirs).
    raw_shared_env = (
        shared_dir_env
        if shared_dir_env is not None
        else str(os.environ.get("SUBHKL_SHARED_DIR") or "").strip()
    )
    if raw_shared_env and not raw_shared_env.startswith("$"):
        env_srv, env_path = split_network_path(raw_shared_env)
        if env_srv is None and (
            not use_real_fs
            or os.path.isdir(env_path)
            or os.path.isdir(os.path.dirname(env_path))
        ):
            env_net = to_shared_network_path(env_path, mountinfo_text=mountinfo_text)
            _, env_net_path = split_network_path(env_net)
            for base_prefix in {
                posixpath.normpath(env_path),
                posixpath.normpath(env_net_path),
                _normalize_ornl_export_path(env_srv, env_net_path),
            }:
                for cand_exp in cand_exports:
                    if _is_posix_prefix(cand_exp, base_prefix):
                        tail = _relative_posix_tail(cand_exp, base_prefix)
                        mapped = (
                            posixpath.normpath(posixpath.join(env_path, tail))
                            if tail
                            else posixpath.normpath(env_path)
                        )
                        if not must_exist or not use_real_fs or os.path.exists(mapped):
                            return mapped

    if must_exist:
        if missing_mount_cand is not None:
            raise FileNotFoundError(
                f"Resolved shared network path '{raw_str}' to local mount '{missing_mount_cand}', "
                "but the file does not exist."
            )
        raise FileNotFoundError(
            f"Could not resolve shared network path '{raw_str}' to an existing local mount point."
        )
    if net_server is None:
        return posixpath.normpath(remote_path)
    raise FileNotFoundError(
        f"Could not map network path '{raw_str}' to any local mount point in /proc/self/mountinfo."
    )


def _chmod_shared_rw(path: str, *, is_dir: bool = False) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o777 if is_dir else 0o666)


def _fsync_directory(dir_path: str | None) -> None:
    """Best-effort ``os.fsync`` on a directory FD so NFS/GPFS commits the new dentry."""
    if not dir_path or not os.path.isdir(dir_path):
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    dfd: int | None = None
    with contextlib.suppress(OSError):
        dfd = os.open(dir_path, flags)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _revalidate_nfs_parent_dir(dir_path: str | None) -> None:
    """Open and close ``dir_path`` to trigger NFS close-to-open directory attribute revalidation."""
    if not dir_path or not os.path.isdir(dir_path):
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    with contextlib.suppress(OSError):
        dfd = os.open(dir_path, flags)
        os.close(dfd)


def get_shared_dir_local(
    *,
    require: bool = False,
    create: bool = True,
    writable: bool | None = None,
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> str | None:
    """Return local mount path for ``SUBHKL_SHARED_DIR`` (raising if ``require=True`` and unset).

    Worker writers pass ``create=True`` (default ``writable=True``) to verify ``W_OK | X_OK``.
    Read-only consumers such as the UI container pass ``create=False, writable=False`` so a
    read-only (``ro``) network mount only requires ``R_OK | X_OK``.
    """
    raw_val = str(
        shared_dir_override
        if shared_dir_override is not None
        else (os.environ.get("SUBHKL_SHARED_DIR") or "")
    ).strip()
    if not raw_val or raw_val.startswith("$"):
        if require:
            raise RuntimeError(
                "SUBHKL_SHARED_DIR is mandatory in galaxy execution mode."
            )
        return None

    local_dir = resolve_shared_local_path(
        raw_val,
        mountinfo_text=mountinfo_text,
        shared_dir_env="",
        must_exist=False,
    )
    need_write = bool(create) if writable is None else bool(writable)
    if mountinfo_text is None:
        if not os.path.isdir(local_dir):
            with contextlib.suppress(OSError):
                os.makedirs(local_dir, mode=0o777, exist_ok=True)
                _chmod_shared_rw(local_dir, is_dir=True)
        if need_write:
            has_write_bits = True
            if (
                hasattr(os, "geteuid")
                and os.geteuid() == 0
                and os.path.isdir(local_dir)
            ):
                with contextlib.suppress(OSError):
                    has_write_bits = bool(os.stat(local_dir).st_mode & 0o222)
            if (
                not os.path.isdir(local_dir)
                or not has_write_bits
                or not os.access(local_dir, os.W_OK | os.X_OK)
            ):
                raise RuntimeError(
                    f"SUBHKL_SHARED_DIR ('{raw_val}' -> '{local_dir}') is not a writable directory."
                )
        elif require and (
            not os.path.isdir(local_dir) or not os.access(local_dir, os.R_OK | os.X_OK)
        ):
            raise RuntimeError(
                f"SUBHKL_SHARED_DIR ('{raw_val}' -> '{local_dir}') is not a readable directory."
            )
    return local_dir


def allocate_shared_artifact_path(
    artifact_name: str = "merged.h5",
    *,
    tag: str = "",
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> tuple[str, str]:
    """Allocate a brand-new UUIDv4 ``(local_path, network_path)`` inside ``SUBHKL_SHARED_DIR``.

    Uses a full 128-bit random UUIDv4 without probing/stat'ing the target path beforehand so
    downstream NFS clients are guaranteed a cold dcache/attribute/page-cache miss on first open.
    """
    _ = tag  # Kept in signature for caller compatibility; human tags live in JSON pointer metadata.
    shared_local = get_shared_dir_local(
        require=True,
        create=(mountinfo_text is None),
        shared_dir_override=shared_dir_override,
        mountinfo_text=mountinfo_text,
    )
    assert shared_local is not None
    clean_art = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_"
        for ch in os.path.basename(artifact_name or "artifact.h5")
    )
    uid = uuid.uuid4().hex
    fname = f"subhkl_shared_{uid}_{clean_art}"
    local_path = posixpath.join(shared_local, fname)
    net_path = to_shared_network_path(local_path, mountinfo_text=mountinfo_text)
    return local_path, net_path


def stage_file_to_shared_dir(
    src_local_path: str | os.PathLike[str],
    artifact_name: str = "merged.h5",
    *,
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> tuple[str, str, str]:
    """Write ``src_local_path`` once to a brand-new UUID path in ``SUBHKL_SHARED_DIR``, ``fsync``, and close.

    Guarantees NFS close-to-open consistency without locking or ``rename``:
      1. Allocates a fresh 128-bit UUIDv4 filename (never previously looked up on any client).
      2. Opens the target exclusively with ``O_WRONLY | O_CREAT | O_EXCL`` and streams all bytes.
      3. Applies ``0666`` mode via ``fchmod``, calls ``os.fsync(fd)`` to force server ``COMMIT``,
         closes ``fd`` completely, and ``fsync``s the parent directory.
    Returns ``(shared_local_path, network_path, artifact_uuid)``.
    """
    src_str = os.fspath(src_local_path)
    if not os.path.isfile(src_str):
        raise FileNotFoundError(
            f"Cannot stage missing local artifact '{src_str}' into SUBHKL_SHARED_DIR."
        )
    expected_bytes = int(os.path.getsize(src_str))
    shared_local, net_path = allocate_shared_artifact_path(
        artifact_name,
        shared_dir_override=shared_dir_override,
        mountinfo_text=mountinfo_text,
    )
    m_uid = _SHARED_UUID_FILENAME_RE.match(os.path.basename(shared_local))
    artifact_uuid = m_uid.group(1) if m_uid else uuid.uuid4().hex

    out_fd: int | None = None
    succeeded = False
    try:
        out_fd = os.open(
            shared_local,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        written = 0
        with open(src_str, "rb") as src_fh:
            while True:
                chunk = src_fh.read(_STAGE_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                mv = memoryview(chunk)
                off = 0
                while off < len(mv):
                    n_put = os.write(out_fd, mv[off:])
                    if n_put <= 0:
                        raise OSError(
                            f"Short write while staging '{shared_local}' to SUBHKL_SHARED_DIR."
                        )
                    off += n_put
                    written += n_put
        if written != expected_bytes:
            raise OSError(
                f"Incomplete write to '{shared_local}': wrote {written} of {expected_bytes} bytes."
            )
        if hasattr(os, "fchmod"):
            with contextlib.suppress(OSError):
                os.fchmod(out_fd, 0o666)
        os.fsync(out_fd)
        fd_to_close = out_fd
        out_fd = None
        os.close(fd_to_close)
        succeeded = True
    finally:
        if out_fd is not None:
            with contextlib.suppress(OSError):
                os.close(out_fd)
        if not succeeded:
            with contextlib.suppress(OSError):
                os.unlink(shared_local)

    _fsync_directory(os.path.dirname(shared_local))
    return shared_local, net_path, artifact_uuid


def read_shared_pointer(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """If ``path`` is a SubHKL shared-storage JSON pointer file, return its parsed dict; else ``None``."""
    if not path:
        return None
    p_str = os.fspath(path)
    if not os.path.isfile(p_str):
        return None
    try:
        st_size = os.path.getsize(p_str)
        if st_size <= 2 or st_size > _MAX_POINTER_FILE_BYTES:
            return None
        with open(p_str, "rb") as fh:
            head = fh.read(8)
            if head.startswith(b"\x89HDF\r\n\x1a\n"):
                return None
            if not head.lstrip().startswith(b"{"):
                return None
            raw_bytes = head + fh.read()
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return None
    if (
        isinstance(doc, dict)
        and doc.get(SHARED_POINTER_MAGIC_KEY) == SHARED_POINTER_VERSION
        and isinstance(doc.get("network_path"), str)
        and doc["network_path"].strip()
    ):
        return doc
    return None


def is_shared_pointer_file(path: str | os.PathLike[str] | None) -> bool:
    """Return True if ``path`` is a valid SubHKL shared-storage JSON pointer file."""
    return read_shared_pointer(path) is not None


def write_shared_pointer(
    pointer_path: str | os.PathLike[str],
    local_shared_h5_path: str | os.PathLike[str],
    *,
    artifact: str = "merged.h5",
    delete_after_use: bool = False,
    metadata: dict[str, Any] | None = None,
    mountinfo_text: str | None = None,
) -> dict[str, Any]:
    """Publish a compact JSON pointer file AFTER ``local_shared_h5_path`` is closed and ``fsync``ed."""
    local_str = os.fspath(local_shared_h5_path)
    if mountinfo_text is None and not os.path.isfile(local_str):
        raise FileNotFoundError(
            f"Refusing to publish shared pointer before '{local_str}' exists on disk."
        )
    net_path = to_shared_network_path(local_str, mountinfo_text=mountinfo_text)
    shared_dir_net = to_shared_network_path(
        os.path.dirname(os.path.abspath(local_str)),
        mountinfo_text=mountinfo_text,
    )
    m_uid = _SHARED_UUID_FILENAME_RE.match(os.path.basename(local_str))
    doc: dict[str, Any] = {
        SHARED_POINTER_MAGIC_KEY: SHARED_POINTER_VERSION,
        "artifact": str(artifact),
        "network_path": net_path,
        "shared_dir_network_path": shared_dir_net,
        "delete_after_use": bool(delete_after_use),
        "created_at": time.time(),
    }
    if m_uid:
        doc["artifact_uuid"] = m_uid.group(1)
    if mountinfo_text is None and os.path.isfile(local_str):
        with contextlib.suppress(OSError):
            doc["size_bytes"] = int(os.path.getsize(local_str))
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if k not in doc and v is not None:
                doc[k] = v

    payload_bytes = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ptr_str = os.fspath(pointer_path)
    parent = os.path.dirname(os.path.abspath(ptr_str))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ptr_str, "wb") as fh:
        fh.write(payload_bytes)
        fh.flush()
        os.fsync(fh.fileno())
    _chmod_shared_rw(ptr_str, is_dir=False)
    _fsync_directory(parent)
    return doc


def publish_shared_artifact(
    src_local_path: str | os.PathLike[str],
    pointer_path: str | os.PathLike[str],
    *,
    artifact: str = "merged.h5",
    delete_after_use: bool = False,
    metadata: dict[str, Any] | None = None,
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Stage ``src_local_path`` to a unique UUID file, ``fsync`` + close it, then publish ``pointer_path``."""
    shared_local, _net_path, uid = stage_file_to_shared_dir(
        src_local_path,
        artifact_name=artifact,
        shared_dir_override=shared_dir_override,
        mountinfo_text=mountinfo_text,
    )
    merged_meta = dict(metadata or {})
    merged_meta.setdefault("artifact_uuid", uid)
    ptr_doc = write_shared_pointer(
        pointer_path,
        shared_local,
        artifact=artifact,
        delete_after_use=delete_after_use,
        metadata=merged_meta,
        mountinfo_text=mountinfo_text,
    )
    return shared_local, ptr_doc


def resolve_input_h5_path(
    path_or_pointer: str | os.PathLike[str],
    *,
    mountinfo_text: str | None = None,
    must_exist: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve an input CLI/tool path that may be a JSON shared pointer, network path, or local HDF5."""
    raw_str = os.fspath(path_or_pointer).strip()
    ptr_doc = read_shared_pointer(raw_str) if mountinfo_text is None else None
    if ptr_doc is None and mountinfo_text is not None and os.path.isfile(raw_str):
        ptr_doc = read_shared_pointer(raw_str)
    if ptr_doc is not None:
        resolved = resolve_shared_local_path(
            ptr_doc["network_path"],
            mountinfo_text=mountinfo_text,
            must_exist=must_exist,
        )
        return resolved, ptr_doc

    srv, _ = split_network_path(raw_str)
    if srv is not None or (
        mountinfo_text is not None
        or (not os.path.exists(raw_str) and raw_str.startswith(("/gpfs/", "/scratch/")))
    ):
        with contextlib.suppress(FileNotFoundError, ValueError):
            resolved_net = resolve_shared_local_path(
                raw_str,
                mountinfo_text=mountinfo_text,
                must_exist=must_exist,
            )
            return resolved_net, None
        if srv is not None and must_exist:
            resolve_shared_local_path(
                raw_str,
                mountinfo_text=mountinfo_text,
                must_exist=True,
            )

    return raw_str, None


def is_safe_shared_intermediate(
    local_path: str | os.PathLike[str],
    *,
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> bool:
    """Verify ``local_path`` is a SubHKL intermediate inside ``SUBHKL_SHARED_DIR`` (never raw /SNS or /HFIR)."""
    abs_p = posixpath.normpath(os.path.abspath(os.fspath(local_path)))
    if abs_p.startswith(("/SNS/", "/HFIR/")) or "/nexus/" in abs_p:
        return False
    bn = os.path.basename(abs_p)
    if bn.startswith("subhkl_shared_"):
        return True
    shared_root = get_shared_dir_local(
        require=False,
        create=False,
        shared_dir_override=shared_dir_override,
        mountinfo_text=mountinfo_text,
    )
    if not shared_root:
        return False
    norm_root = posixpath.normpath(os.path.abspath(shared_root))
    return abs_p != norm_root and _is_posix_prefix(abs_p, norm_root)


def cleanup_shared_artifact(
    pointer_or_path: str | os.PathLike[str] | None,
    *,
    only_if_delete_after_use: bool = False,
    shared_dir_override: str | None = None,
    mountinfo_text: str | None = None,
) -> bool:
    """Delete a shared intermediate file (given either its JSON pointer path or direct network/local path)."""
    if not pointer_or_path:
        return False
    raw_str = os.fspath(pointer_or_path).strip()
    if not raw_str:
        return False

    ptr_doc = read_shared_pointer(raw_str)
    if ptr_doc is not None:
        if only_if_delete_after_use and not bool(ptr_doc.get("delete_after_use")):
            return False
        target_spec = str(ptr_doc["network_path"])
    else:
        if only_if_delete_after_use:
            return False
        target_spec = raw_str

    try:
        local_target = resolve_shared_local_path(
            target_spec,
            mountinfo_text=mountinfo_text,
            shared_dir_env=shared_dir_override,
            must_exist=False,
        )
    except (FileNotFoundError, ValueError):
        return False

    if not is_safe_shared_intermediate(
        local_target,
        shared_dir_override=shared_dir_override,
        mountinfo_text=mountinfo_text,
    ):
        return False

    if Path(local_target).is_file() or Path(local_target).is_symlink():
        with contextlib.suppress(OSError):
            os.unlink(local_target)
            return not os.path.exists(local_target)
    return False
