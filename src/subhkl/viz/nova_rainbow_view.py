"""NOVA / Trame and NDIP Galaxy Interactive Tool integration for Rainbow Zones.

Supports three execution modes for NDIP / Galaxy / NOVA and local beamline workstations:
1. **NOVA / Trame View (`RainbowZonesTrameApp`)**: When ``trame`` (and optionally
   ``nova-trame``) is installed inside the container, mounts the Rainbow Zones
   interactive view inside a Trame/Vuetify server with bidirectional state sync.
2. **Built-in HTTP Interactive Server (`serve_rainbow_zones`)**: Zero-dependency
   HTTP server on ``0.0.0.0:8080`` (honoring ``$EP_PATH`` for Galaxy's
   ``gx-it-proxy`` / nginx reverse proxy) so the ``ghcr.io/neutrons/subhkl``
   container can run as a Galaxy Interactive Tool without requiring extra packages.
3. **Standalone HTML Export**: Produces a self-contained HTML artifact for
   Galaxy dataset history viewing.
"""

from __future__ import annotations

import atexit
import contextlib
import errno
import http.client
import http.server
import io
import json
import math
import os
import re
import shutil
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

for _thr_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thr_var, "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from subhkl import get_container_image
from subhkl.viz.logo_svg import SUBHKL_FAVICON_DATA_URI, SUBHKL_FAVICON_SVG
from subhkl.viz.rainbow_zones import (
    build_rainbow_zone_payload,
    render_rainbow_html,
)

_NOVA_RECOVERABLE_ERRORS = (
    OSError,
    http.client.HTTPException,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    ImportError,
    ArithmeticError,
)

# Datasets at or above this size on remote Galaxy servers use HTTP Range + HDF5
# object-copy to sync 2D detector images & metadata locally while skipping multi-GB
# ``events/tof`` arrays (which stay in Galaxy's object store for Pulsar GPU jobs).
_SLIM_REMOTE_H5_MIN_BYTES = 32 * 1024 * 1024
_HTTP_RANGE_BLOCK_BYTES = 8 * 1024 * 1024


class _RangeFallbackError(RuntimeError):
    """Internal signal to fall back from HTTP Range HDF5 reading to sequential streaming."""


class _HttpRangeReader(io.RawIOBase):
    """Seekable read-only file-like wrapper over HTTP ``Range`` requests with LRU block caching."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        total_size: int,
        initial_block: bytes,
        block_size: int | None = None,
        connect_timeout: float = 20.0,
        read_timeout: float = 60.0,
        max_fetch_bytes: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        super().__init__()
        import requests

        self._url = url
        self._base_headers = dict(headers)
        self.size = int(total_size)
        self.pos = 0
        self.block_size = max(
            65536, int(_HTTP_RANGE_BLOCK_BYTES if block_size is None else block_size)
        )
        self._connect_timeout = float(connect_timeout)
        self._read_timeout = float(read_timeout)
        self.max_fetch_bytes = (
            int(max_fetch_bytes)
            if max_fetch_bytes is not None
            else max(512 * 1024 * 1024, int(0.40 * self.size))
        )
        self._on_progress = on_progress
        self._session = requests.Session()
        self._cache: dict[int, bytes] = {}
        self.n_requests = 1 if initial_block else 0
        self.bytes_fetched = len(initial_block)
        if initial_block and len(initial_block) == min(self.size, self.block_size):
            self._cache[0] = initial_block

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = int(offset)
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + int(offset)
        elif whence == io.SEEK_END:
            new_pos = self.size + int(offset)
        else:
            raise ValueError(f"Unsupported whence={whence}")
        if new_pos < 0:
            raise OSError(errno.EINVAL, "Negative seek offset")
        self.pos = new_pos
        return self.pos

    def _fetch_block(self, blk_idx: int) -> bytes:
        cached = self._cache.get(blk_idx)
        if cached is not None:
            # Refresh LRU order
            self._cache.pop(blk_idx, None)
            self._cache[blk_idx] = cached
            return cached
        start = blk_idx * self.block_size
        if start >= self.size:
            return b""
        end = min(self.size - 1, start + self.block_size - 1)
        expected_len = end - start + 1
        if self.bytes_fetched + expected_len > self.max_fetch_bytes:
            raise _RangeFallbackError(
                f"HTTP Range fetch exceeded budget ({self.bytes_fetched} / {self.max_fetch_bytes} bytes)"
            )
        req_headers = dict(self._base_headers)
        req_headers["Range"] = f"bytes={start}-{end}"
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._session.get(
                    self._url,
                    headers=req_headers,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
                try:
                    if resp.status_code != 206:
                        raise _RangeFallbackError(
                            f"Expected HTTP 206 for Range {start}-{end}, got {resp.status_code}"
                        )
                    data = resp.content
                finally:
                    resp.close()
                if not data:
                    raise OSError(f"Empty HTTP 206 body for Range {start}-{end}")
                self.n_requests += 1
                self.bytes_fetched += len(data)
                if len(self._cache) >= 12:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[blk_idx] = data
                if callable(self._on_progress):
                    self._on_progress(self.bytes_fetched, self.size)
                return data
            except _RangeFallbackError:
                raise
            except (OSError, http.client.HTTPException) as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.4)
        assert last_exc is not None
        raise last_exc

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        mem = memoryview(b).cast("B")
        n_req = min(len(mem), self.size - self.pos)
        if n_req <= 0:
            return 0
        written = 0
        while written < n_req:
            cur = self.pos + written
            blk_idx = cur // self.block_size
            blk_off = cur % self.block_size
            blk = self._fetch_block(blk_idx)
            if blk_off >= len(blk):
                break
            take = min(n_req - written, len(blk) - blk_off)
            if take <= 0:
                break
            mem[written : written + take] = blk[blk_off : blk_off + take]
            written += take
        self.pos += written
        return written

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._session.close()
        self._cache.clear()
        super().close()


_TEMP_CLEANUP_LOCK = threading.Lock()
_REGISTERED_TEMP_PATHS: dict[str, str] = {}
_REGISTERED_SHARED_ARTIFACTS: dict[str, dict[str, Any]] = {}
_ATEXIT_CLEANUP_REGISTERED = False
_SIGNAL_CLEANUP_INSTALLED = False


def _is_strictly_inside_root(path_str: str, root_str: str) -> bool:
    """Return True only if path_str is a strict descendant of root_str."""
    if not path_str or not root_str:
        return False
    abs_p = os.path.abspath(path_str)
    abs_r = os.path.abspath(root_str)
    if abs_p == abs_r:
        return False
    try:
        if os.path.commonpath([abs_p, abs_r]) != abs_r:
            return False
        real_parent = os.path.realpath(os.path.dirname(abs_p))
        real_r = os.path.realpath(abs_r)
        return os.path.commonpath([real_parent, real_r]) == real_r
    except (ValueError, OSError):
        return False


_MAX_NDIP_COMPACT_UPLOAD_BYTES = 50 * 1024 * 1024


def _chmod_world_rw(
    path: str | os.PathLike[str] | None,
    *,
    is_dir: bool | None = None,
) -> None:
    """Best-effort chmod (0o777 dirs, 0o666 files) so Pulsar host cleanup never hits EACCES."""
    if not path:
        return
    p_str = os.fspath(path)
    with contextlib.suppress(OSError):
        if os.path.islink(p_str):
            return
        dir_flag = os.path.isdir(p_str) if is_dir is None else bool(is_dir)
        os.chmod(p_str, 0o777 if dir_flag else 0o666)


def _get_staging_root() -> str:
    """Return local NVMe staging directory (``SUBHKL_STAGING_DIR`` if set, else ``TMPDIR``)."""
    raw_env = str(os.environ.get("SUBHKL_STAGING_DIR") or "").strip()
    if raw_env and not raw_env.startswith("$"):
        abs_env = os.path.abspath(raw_env)
        with contextlib.suppress(OSError):
            os.makedirs(abs_env, mode=0o777, exist_ok=True)
            _chmod_world_rw(abs_env, is_dir=True)
        if os.path.isdir(abs_env) and os.access(abs_env, os.W_OK | os.X_OK):
            return abs_env
    return os.path.abspath(tempfile.gettempdir())


def _is_readonly_galaxy_dat(path: str | os.PathLike[str] | None) -> bool:
    """Return True if ``path`` looks like a read-only Galaxy/Pulsar ``dataset_*.dat`` input."""
    if not path:
        return False
    abs_p = os.path.abspath(os.fspath(path))
    bn = os.path.basename(abs_p)
    if bn.startswith("dataset_") and bn.endswith(".dat"):
        return True
    if "/staging" in abs_p:
        staging_root = _get_staging_root()
        if _is_strictly_inside_root(abs_p, staging_root):
            return False
        return not ("subhkl_work_" in abs_p or "subhkl_staging" in abs_p)
    return False


def _matching_temp_root(abs_p: str) -> str | None:
    """Return the enclosing staging or TMPDIR root for ``abs_p``, or ``None``."""
    roots = [_get_staging_root(), os.path.abspath(tempfile.gettempdir())]
    seen: set[str] = set()
    for cand_root in roots:
        if cand_root in seen:
            continue
        seen.add(cand_root)
        if _is_strictly_inside_root(abs_p, cand_root):
            return cand_root
    return None


def register_temp_artifact(path: str | os.PathLike[str] | None) -> str | None:
    """Track an app-created file or directory inside staging/TMPDIR for removal at process exit."""
    global _ATEXIT_CLEANUP_REGISTERED
    if not path:
        return None
    abs_p = os.path.abspath(os.fspath(path))
    tmp_root = _matching_temp_root(abs_p)
    if not tmp_root:
        return None
    with _TEMP_CLEANUP_LOCK:
        _REGISTERED_TEMP_PATHS[abs_p] = tmp_root
        if not _ATEXIT_CLEANUP_REGISTERED:
            atexit.register(cleanup_registered_temp_artifacts)
            _ATEXIT_CLEANUP_REGISTERED = True
    return abs_p


def _remove_temp_artifact_now(path: str | os.PathLike[str] | None) -> bool:
    """Immediately delete a single tracked staging/TMPDIR file or directory if safe."""
    if not path:
        return False
    abs_p = os.path.abspath(os.fspath(path))
    with _TEMP_CLEANUP_LOCK:
        recorded_root = _REGISTERED_TEMP_PATHS.pop(abs_p, None)
    tmp_root = recorded_root or _matching_temp_root(abs_p)
    if not tmp_root or not _is_strictly_inside_root(abs_p, tmp_root):
        return False
    with contextlib.suppress(OSError):
        if os.path.islink(abs_p) or os.path.isfile(abs_p):
            os.unlink(abs_p)
            return True
        if os.path.isdir(abs_p):
            shutil.rmtree(abs_p, ignore_errors=True)
            return not os.path.exists(abs_p)
    return False


def register_shared_intermediate(
    path: str | os.PathLike[str] | None,
    *,
    network_path: str = "",
    dataset_id: str = "",
    history_id: str = "",
    work_dir: str = "",
    delete_after_use: bool = False,
) -> str | None:
    """Track a SUBHKL_SHARED_DIR intermediate file for lifecycle and process-exit cleanup."""
    global _ATEXIT_CLEANUP_REGISTERED
    if not path:
        return None
    from subhkl.io.shared_storage import (
        is_safe_shared_intermediate,
        to_shared_network_path,
    )

    real_p = os.path.abspath(os.path.realpath(os.fspath(path)))
    if not is_safe_shared_intermediate(real_p):
        return None
    net_p = network_path or to_shared_network_path(real_p)
    with _TEMP_CLEANUP_LOCK:
        prev = _REGISTERED_SHARED_ARTIFACTS.get(real_p, {})
        _REGISTERED_SHARED_ARTIFACTS[real_p] = {
            "local_path": real_p,
            "network_path": net_p,
            "dataset_id": str(dataset_id or prev.get("dataset_id") or ""),
            "history_id": str(history_id or prev.get("history_id") or ""),
            "work_dir": (
                os.path.abspath(work_dir)
                if work_dir
                else str(prev.get("work_dir") or "")
            ),
            "delete_after_use": bool(
                delete_after_use or prev.get("delete_after_use", False)
            ),
        }
        if not _ATEXIT_CLEANUP_REGISTERED:
            atexit.register(cleanup_registered_temp_artifacts)
            _ATEXIT_CLEANUP_REGISTERED = True
    return real_p


def _symlink_shared_into_work_dir(src_shared_path: str, dest_work_path: str) -> None:
    """Create a zero-copy symlink in local NVMe work_dir pointing to a shared intermediate."""
    abs_src = os.path.abspath(src_shared_path)
    abs_dest = os.path.abspath(dest_work_path)
    if abs_src == abs_dest:
        return
    if os.path.islink(abs_dest):
        with contextlib.suppress(OSError):
            if os.path.abspath(os.path.realpath(abs_dest)) == abs_src:
                return
    with contextlib.suppress(OSError):
        if os.path.lexists(abs_dest) and not os.path.isdir(abs_dest):
            os.unlink(abs_dest)
        os.makedirs(os.path.dirname(abs_dest), mode=0o777, exist_ok=True)
        os.symlink(abs_src, abs_dest)


def cleanup_registered_temp_artifacts() -> list[str]:
    """Remove all tracked local TMPDIR/staging files and SUBHKL_SHARED_DIR intermediates."""
    from subhkl.io.shared_storage import cleanup_shared_artifact

    with _TEMP_CLEANUP_LOCK:
        items = list(_REGISTERED_TEMP_PATHS.items())
        _REGISTERED_TEMP_PATHS.clear()
        shared_items = list(_REGISTERED_SHARED_ARTIFACTS.values())
        _REGISTERED_SHARED_ARTIFACTS.clear()
    removed: list[str] = []
    for sh_meta in shared_items:
        sh_local = str(sh_meta.get("local_path") or "")
        if sh_local and cleanup_shared_artifact(sh_local):
            removed.append(sh_local)
    for abs_p, tmp_root in sorted(items, key=lambda kv: len(kv[0]), reverse=True):
        if not _is_strictly_inside_root(abs_p, tmp_root):
            continue
        with contextlib.suppress(OSError):
            if os.path.islink(abs_p) or os.path.isfile(abs_p):
                os.unlink(abs_p)
                removed.append(abs_p)
            elif os.path.isdir(abs_p):
                shutil.rmtree(abs_p, ignore_errors=True)
                if not os.path.exists(abs_p):
                    removed.append(abs_p)
    return removed


def install_exit_cleanup_signal_handlers(
    on_shutdown: Callable[[], None] | None = None,
) -> None:
    """Ensure SIGTERM cleanly triggers active-job cancellation and TMPDIR cleanup."""
    global _SIGNAL_CLEANUP_INSTALLED
    if threading.current_thread() is not threading.main_thread():
        return
    with _TEMP_CLEANUP_LOCK:
        if _SIGNAL_CLEANUP_INSTALLED:
            return
        _SIGNAL_CLEANUP_INSTALLED = True
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, frame: Any) -> None:
        if callable(on_shutdown):
            with contextlib.suppress(Exception):
                on_shutdown()
        cleanup_registered_temp_artifacts()
        if callable(prev_sigterm) and prev_sigterm not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            prev_sigterm(signum, frame)
        raise SystemExit(128 + signum)

    with contextlib.suppress(OSError, ValueError):
        signal.signal(signal.SIGTERM, _handle_sigterm)


class NovaGalaxyPipelineRunner:
    """Orchestrates NDIP / Galaxy IMAGINE-X diffraction workflow steps.

    Supports two execution topologies:
      1. **Local-staging GPU app mode** (``SUBHKL_EXEC_MODE=local``, or automatically inside
         an NDIP interactive GPU tool session with ``EP_PATH`` and ``/SNS`` / ``/HFIR`` mounted):
         - ``Reduce``, ``Solve``, and ``Integrate`` execute directly inside the app's local
           NVMe staging directory (``SUBHKL_STAGING_DIR`` / ``TMPDIR``), keeping multi-GB
           ``merged.h5`` / ``integrated.h5`` event and image arrays off Rucio/Ceph object storage.
         - Only compact MB-scale deliverables (``solve.h5`` metadata, ``reflections.mtz``,
           merged ``.mtz`` files, and merge tables/JSON) are uploaded to NDIP Galaxy History.
         - Stage 4 (``Merge`` / Careless) still routes through NDIP Galaxy
           (``imaginex_subhkl_merge``) whenever ``careless`` is not installed locally, exchanging
           only compact ``.mtz`` files.
      2. **Remote Galaxy tool mode** (``SUBHKL_EXEC_MODE=galaxy``, or thin client without local
         facility mounts): dispatches all four stages via ``nova-galaxy`` -> Galaxy -> Pulsar.
    """

    STAGE_TOOLS: ClassVar[dict[str, list[str]]] = {
        "reduce": [
            "imaginex_subhkl_prepare_peaks",
        ],
        "solve": ["imaginex_subhkl_solve"],
        "integrate": [
            "imaginex_subhkl_predict",
            "imaginex_subhkl_integrate",
        ],
        "merge": ["imaginex_subhkl_merge"],
    }

    # Upstream NDIP ``neutrons_load_data`` tool hardcodes ``<select name="instrument">``
    # options when ``params|ingest_mode == "oncat"``, omitting instruments such as
    # ``CG4D`` / ``IMAGINE``. When an instrument is not in this allow-list, we pass
    # the ONCat-resolved path via ``params|ingest_mode == "file"`` (``params|filepath``).
    NEUTRONS_LOAD_DATA_ONCAT_INSTRUMENTS: ClassVar[dict[str, set[str]]] = {
        "SNS": {
            "ARCS",
            "BL0",
            "CNCS",
            "CORELLI",
            "EQSANS",
            "FNPB",
            "HYSPEC",
            "MANDI",
            "NOMAD",
            "NSE",
            "PG3",
            "POWGEN",
            "REF_L",
            "REF_M",
            "SEQ",
            "SNAP",
            "TOPAZ",
            "USANS",
            "VENUS",
            "VISION",
            "VULCAN",
        },
        "HFIR": {
            "CG1B",
            "CG1D",
            "CG2",
            "CG3",
            "CG4C",
            "HB1",
            "HB1A",
            "HB2A",
            "HB2B",
            "HB2C",
            "HB3",
            "HB3A",
        },
    }

    def __init__(
        self,
        galaxy_url: str | None = None,
        galaxy_api_key: str | None = None,
        history_id: str | None = None,
    ) -> None:
        raw_url = galaxy_url or os.environ.get("GALAXY_URL") or ""
        if raw_url.startswith("$"):
            raw_url = ""
        raw_key = galaxy_api_key or os.environ.get("GALAXY_API_KEY") or ""
        if raw_key.startswith("$"):
            raw_key = ""
        raw_hid = history_id or os.environ.get("HISTORY_ID") or ""
        if raw_hid.startswith("$"):
            raw_hid = ""

        self.galaxy_url = raw_url.rstrip("/") if raw_url else None
        self.galaxy_api_key = raw_key or None
        self.history_id = raw_hid or None
        self._explicit_history_id: bool = bool(
            history_id and not str(history_id).startswith("$")
        )
        self._launch_history_id: str | None = raw_hid or None
        self._subhkl_datastore_history_id: str | None = None
        self._probed_subhkl_history_ts: float = 0.0
        self._dataset_history_by_id: dict[str, str] = {}
        self._conn_helper: Any = None
        self.active_job: dict[str, Any] | None = None
        self.step_timings: dict[str, dict[str, Any]] = {}
        self.current_frame_index: int | None = None
        self._upload_lock = threading.RLock()
        self._dataset_id_by_path: dict[str, str] = {}
        self._dataset_sig_by_path: dict[str, tuple[int, int, str]] = {}
        self._auto_discover_local_galaxy()
        with contextlib.suppress(ImportError):
            from subhkl.viz.rainbow_zones import register_remote_tof_slab_loader

            register_remote_tof_slab_loader(self._load_slim_remote_tof_slab)

    def _sync_job_timing(self, job_record: dict[str, Any] | None) -> None:
        """Update per-step elapsed millisecond timing metadata on ``job_record`` and ``self.step_timings``."""
        if not isinstance(job_record, dict):
            return
        stage_id = str(job_record.get("stage_id") or "").strip()
        if not stage_id:
            return
        now_ts = time.time()
        raw_started = job_record.get("started_at")
        started_at = (
            float(raw_started)
            if isinstance(raw_started, (int, float)) and raw_started > 0
            else now_ts
        )
        job_record["started_at"] = started_at
        st = str(job_record.get("state") or "running")
        if st in ("ok", "error", "deleted", "paused"):
            raw_finished = job_record.get("finished_at")
            finished_at = (
                float(raw_finished)
                if isinstance(raw_finished, (int, float)) and raw_finished >= started_at
                else max(started_at, now_ts)
            )
            job_record["finished_at"] = finished_at
            elapsed_ms = round(max(0.0, (finished_at - started_at) * 1000.0))
        else:
            job_record.pop("finished_at", None)
            finished_at = None
            elapsed_ms = round(max(0.0, (now_ts - started_at) * 1000.0))
        job_record["elapsed_ms"] = elapsed_ms
        self.step_timings[stage_id] = {
            "stage_id": stage_id,
            "state": st,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_ms": elapsed_ms,
        }

    def get_step_timings_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of per-stage stopwatch timings (in ms), refreshing any active job."""
        self._sync_job_timing(self.active_job)
        return {k: dict(v) for k, v in self.step_timings.items()}

    def _build_neutrons_load_data_inputs(
        self,
        *,
        ingest_mode: str,
        fac: str,
        inst: str,
        ipts_raw: str,
        run_num: int | None,
        filepath: str,
    ) -> dict[str, Any]:
        """Build ``neutrons_load_data`` inputs, falling back to ``file`` mode for instruments
        (such as HFIR ``CG4D``) omitted from upstream's hardcoded ``oncat`` dropdown."""
        fac_u = str(
            fac or ("SNS" if str(filepath).startswith("/SNS/") else "HFIR")
        ).upper()
        inst_u = str(inst or "").upper()
        allowed_oncat = self.NEUTRONS_LOAD_DATA_ONCAT_INSTRUMENTS.get(fac_u, set())
        if (
            str(ingest_mode or "oncat") != "file"
            and ipts_raw
            and run_num is not None
            and inst_u in allowed_oncat
        ):
            return {
                "params|ingest_mode": "oncat",
                "params|facility": fac_u,
                "params|instrument": inst_u,
                "params|ipts": str(ipts_raw),
                "params|run": int(run_num),
            }
        return {
            "params|ingest_mode": "file",
            "params|filepath": str(filepath),
        }

    def _run_neutrons_load_data_tool(
        self,
        gi: Any,
        history_id: str,
        nld_inputs: dict[str, Any],
        filepath: str,
    ) -> dict[str, Any]:
        """Execute ``neutrons_load_data`` and automatically retry in ``file`` mode if
        upstream Galaxy rejects an ``oncat`` parameter option (HTTP 400)."""
        try:
            return gi.tools.run_tool(
                history_id=history_id,
                tool_id="neutrons_load_data",
                tool_inputs=nld_inputs,
            )
        except Exception:
            if nld_inputs.get("params|ingest_mode") == "oncat" and filepath:
                fallback_inputs = {
                    "params|ingest_mode": "file",
                    "params|filepath": str(filepath),
                }
                nld_inputs.clear()
                nld_inputs.update(fallback_inputs)
                return gi.tools.run_tool(
                    history_id=history_id,
                    tool_id="neutrons_load_data",
                    tool_inputs=nld_inputs,
                )
            raise

    def _lookup_oncat_run_title(self, nxs_path: str) -> str:
        """Look up a NeXus run's ONCat title and goniometer angles without reading the host filesystem."""
        import re as _re

        p_str = str(nxs_path or "")
        m_loc = _re.search(r"/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", p_str, _re.IGNORECASE)
        m_run = _re.search(r"_([0-9]+)\.nxs", os.path.basename(p_str), _re.IGNORECASE)
        if not m_loc:
            return ""
        fac_u = m_loc.group(1).upper()
        inst_u = m_loc.group(2).upper()
        ipts_num = m_loc.group(3)
        run_num = int(m_run.group(1)) if m_run else None
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            cat = self.browse_oncat_catalog(
                {
                    "facility": fac_u,
                    "instrument": inst_u,
                    "ipts": f"IPTS-{ipts_num}",
                }
            )
            for r in cat.get("runs") or []:
                if str(r.get("location") or "") == p_str or (
                    run_num is not None and r.get("run_number") == run_num
                ):
                    t_part = str(r.get("title") or "").strip()
                    g_part = str(r.get("angles") or "").strip()
                    return f"{t_part} {g_part}".strip()
        return ""

    def apply_active_solve_overlay(
        self,
        payload: dict[str, Any],
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """If a solve job is currently active and has produced an intermediate blind solution,
        overlay its zones and ladder steps projected for ``frame_index`` onto ``payload``."""
        if frame_index is not None:
            self.current_frame_index = int(frame_index)
        target_frame = int(
            frame_index if frame_index is not None else (self.current_frame_index or 0)
        )
        aj = self.active_job
        if not isinstance(aj, dict):
            return payload
        if aj.get("stage_id") != "solve" or aj.get("state") not in (
            "queued",
            "running",
        ):
            return payload
        partial_h5 = aj.get("partial_h5_path")
        images_h5 = aj.get("images_h5_path")
        sp = aj.get("solve_params") or {}
        if sp.get("wavelength_range") and len(sp["wavelength_range"]) >= 2:
            payload["wavelength_range"] = [
                float(sp["wavelength_range"][0]),
                float(sp["wavelength_range"][1]),
            ]
        if sp.get("d_min"):
            payload["d_min"] = float(sp["d_min"])
        if (
            partial_h5
            and os.path.isfile(partial_h5)
            and os.path.getsize(partial_h5) > 0
        ):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                from subhkl.viz.rainbow_zones import build_rainbow_zone_payload

                raw_dmin_live = (
                    float(sp["d_min"])
                    if sp.get("d_min")
                    else (float(payload["d_min"]) if payload.get("d_min") else None)
                )
                # Cap intermediate live-solve preview reflections to >= 2.0 A on the
                # active frame so multi-run high-resolution solves stay responsive.
                live_dmin = (
                    max(2.0, raw_dmin_live)
                    if raw_dmin_live is not None and raw_dmin_live > 0
                    else None
                )
                partial_mtime_ns = int(os.stat(partial_h5).st_mtime_ns)
                prev_ov = aj.get("partial_overlay")
                if (
                    isinstance(prev_ov, dict)
                    and prev_ov.get("mtime_ns") == partial_mtime_ns
                    and prev_ov.get("frame_index") == target_frame
                    and prev_ov.get("zones")
                ):
                    payload["zones"] = prev_ov["zones"]
                    if prev_ov.get("peaks"):
                        payload["peaks"] = prev_ov["peaks"]
                    if prev_ov.get("sample_peaks"):
                        payload["sample_peaks"] = prev_ov["sample_peaks"]
                    if prev_ov.get("panels") and payload.get("panels"):
                        p_by_b = {
                            int(p["bank"]): p for p in prev_ov["panels"] if "bank" in p
                        }
                        for cur_p in payload["panels"]:
                            upd_p = p_by_b.get(int(cur_p.get("bank", -1)))
                            if upd_p:
                                for k_g in (
                                    "az_min",
                                    "az_max",
                                    "y_min",
                                    "y_max",
                                    "el_min",
                                    "el_max",
                                    "corners",
                                    "corners_sph",
                                    "outline",
                                    "outline_sph",
                                    "mesh",
                                    "mesh_sph",
                                ):
                                    if k_g in upd_p:
                                        cur_p[k_g] = upd_p[k_g]
                    if prev_ov.get("cell"):
                        payload["cell"] = prev_ov["cell"]
                    if prev_ov.get("ub_matrix"):
                        payload["ub_matrix"] = prev_ov["ub_matrix"]
                    if prev_ov.get("goniometer_R"):
                        payload["goniometer_R"] = prev_ov["goniometer_R"]
                    if prev_ov.get("ladder_steps"):
                        payload["ladder_steps"] = prev_ov["ladder_steps"]
                    payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
                    return payload
                p_blind = build_rainbow_zone_payload(
                    replay_h5=partial_h5,
                    images_h5=(
                        images_h5 if (images_h5 and os.path.isfile(images_h5)) else None
                    ),
                    instrument=str(
                        sp.get("instrument") or payload.get("instrument") or "CG4D"
                    ),
                    space_group=str(
                        sp.get("space_group") or payload.get("space_group") or "P 1"
                    ),
                    cell=None,
                    wavelength_range=tuple(
                        sp.get("wavelength_range")
                        or payload.get("wavelength_range")
                        or (2.0, 4.5)
                    ),
                    d_min=live_dmin,
                    max_zone_index=3,
                    max_zones=60,
                    include_panels=True,
                    include_panel_images=False,
                    include_peaks=True,
                    include_all_runs_peaks=False,
                    n_arc=720,
                    frame_index=target_frame,
                )
                if p_blind.get("zones"):
                    payload["zones"] = p_blind["zones"]
                    if p_blind.get("peaks"):
                        payload["peaks"] = p_blind["peaks"]
                    if p_blind.get("sample_peaks"):
                        payload["sample_peaks"] = p_blind["sample_peaks"]
                    if p_blind.get("panels") and payload.get("panels"):
                        p_by_b = {
                            int(p["bank"]): p for p in p_blind["panels"] if "bank" in p
                        }
                        for cur_p in payload["panels"]:
                            upd_p = p_by_b.get(int(cur_p.get("bank", -1)))
                            if upd_p:
                                for k_g in (
                                    "az_min",
                                    "az_max",
                                    "y_min",
                                    "y_max",
                                    "el_min",
                                    "el_max",
                                    "corners",
                                    "corners_sph",
                                    "outline",
                                    "outline_sph",
                                    "mesh",
                                    "mesh_sph",
                                ):
                                    if k_g in upd_p:
                                        cur_p[k_g] = upd_p[k_g]
                    if p_blind.get("cell"):
                        payload["cell"] = p_blind["cell"]
                    if p_blind.get("ub_matrix"):
                        payload["ub_matrix"] = p_blind["ub_matrix"]
                    if p_blind.get("goniometer_R"):
                        payload["goniometer_R"] = p_blind["goniometer_R"]
                    if p_blind.get("ladder_steps"):
                        payload["ladder_steps"] = p_blind["ladder_steps"]
                    payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
                    next_seq = (
                        int(prev_ov.get("seq", 0)) + 1
                        if isinstance(prev_ov, dict)
                        else 1
                    )
                    aj["partial_overlay"] = {
                        "seq": next_seq,
                        "mtime_ns": partial_mtime_ns,
                        "frame_index": target_frame,
                        "zones": p_blind.get("zones", []),
                        "peaks": p_blind.get("peaks", []),
                        "sample_peaks": p_blind.get("sample_peaks", []),
                        "panels": p_blind.get("panels", []),
                        "cell": p_blind.get("cell"),
                        "ub_matrix": p_blind.get("ub_matrix"),
                        "goniometer_R": p_blind.get("goniometer_R"),
                        "ladder_steps": p_blind.get("ladder_steps", []),
                    }
                    return payload
        ov = aj.get("partial_overlay")
        if (
            isinstance(ov, dict)
            and ov.get("zones")
            and not payload.get("zones")
            and (
                ov.get("frame_index") is None or int(ov["frame_index"]) == target_frame
            )
        ):
            payload["zones"] = ov["zones"]
            if ov.get("peaks"):
                payload["peaks"] = ov["peaks"]
            if ov.get("sample_peaks"):
                payload["sample_peaks"] = ov["sample_peaks"]
            if ov.get("ladder_steps"):
                payload["ladder_steps"] = ov["ladder_steps"]
            payload["active_rung_idx"] = int(aj.get("ladder_rung_idx") or 1)
        return payload

    def sync_frame_index(self, frame_index: int | None) -> None:
        """Update ``current_frame_index`` and ensure any active live solve overlay matches."""
        if frame_index is None:
            return
        self.current_frame_index = int(frame_index)
        aj = self.active_job
        if not isinstance(aj, dict):
            return
        if aj.get("stage_id") != "solve" or aj.get("state") not in (
            "queued",
            "running",
        ):
            return
        ov = aj.get("partial_overlay")
        if isinstance(ov, dict) and ov.get("frame_index") != self.current_frame_index:
            self.apply_active_solve_overlay({}, frame_index=self.current_frame_index)

    @staticmethod
    def _is_galaxy_server(url: str) -> bool:
        """Return True only if ``url`` responds with a valid Galaxy JSON ``/api/version`` payload."""
        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/api/version",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status != 200:
                    return False
                doc = json.loads(resp.read().decode("utf-8"))
                return isinstance(doc, dict) and "version_major" in doc
        except _NOVA_RECOVERABLE_ERRORS:
            return False

    @staticmethod
    def _clean_err_msg(exc: Exception) -> str:
        """Strip any HTML/XML/SVG body or multi-line traceback from an exception message."""
        msg = str(exc).strip()
        for marker in ("<!doctype", "<html", "<?xml", "<svg", "\n"):
            idx = msg.lower().find(marker)
            if idx > 0:
                msg = msg[:idx].strip().rstrip(":,;")
        return msg[:220]

    @staticmethod
    def _format_terminal_stream(raw_text: str) -> str:
        """Emulate terminal carriage-return (\\r) and ANSI line-erase/cursor-up for tqdm progress bars."""
        if not raw_text:
            return ""
        import re as _re

        text = raw_text.replace("\r\n", "\n")
        tokens = _re.split(r"(\x1b\[[0-9;?]*[A-Za-z]|\r|\n)", text)
        lines: list[str] = [""]
        row = 0
        col = 0
        for tok in tokens:
            if not tok:
                continue
            if tok == "\n":
                lines[row] = lines[row].rstrip()
                row += 1
                if row >= len(lines):
                    lines.append("")
                col = 0
            elif tok == "\r":
                col = 0
            elif tok.startswith("\x1b["):
                cmd_ch = tok[-1]
                params_s = tok[2:-1].lstrip("?;")
                first_num = (
                    int(params_s.split(";")[0])
                    if params_s and params_s.split(";")[0].isdigit()
                    else 1
                )
                if cmd_ch == "A":
                    row = max(0, row - max(1, first_num))
                    col = 0
                elif cmd_ch == "B":
                    row = min(len(lines) - 1, row + max(1, first_num))
                    col = 0
                elif cmd_ch == "G":
                    col = max(0, first_num - 1)
                elif cmd_ch == "K":
                    mode_k = int(params_s) if params_s and params_s.isdigit() else 0
                    if mode_k == 2:
                        lines[row] = ""
                        col = 0
                    elif mode_k == 1:
                        lines[row] = (" " * col) + lines[row][col:]
                    else:
                        lines[row] = lines[row][:col]
            else:
                cur = lines[row]
                if col > len(cur):
                    cur = cur + (" " * (col - len(cur)))
                lines[row] = cur[:col] + tok + cur[col + len(tok) :]
                col += len(tok)

        tqdm_re = _re.compile(r"^\s*(?P<pfx>[^%|]*?)\s*\d{1,3}%\|.*\|")
        collapsed: list[str] = []
        for ln in lines:
            clean_ln = ln.rstrip()
            if clean_ln.lstrip().startswith("[subhkl-live-state]"):
                continue
            m_cur = tqdm_re.match(clean_ln)
            if m_cur:
                while collapsed and collapsed[-1] == "":
                    collapsed.pop()
                if collapsed:
                    m_prev = tqdm_re.match(collapsed[-1])
                    if (
                        m_prev
                        and m_prev.group("pfx").strip() == m_cur.group("pfx").strip()
                    ):
                        collapsed[-1] = clean_ln
                        continue
            collapsed.append(clean_ln)
        while (
            len(collapsed) >= 2 and collapsed[-1] == "" and tqdm_re.match(collapsed[-2])
        ):
            collapsed.pop()
        return "\n".join(collapsed)

    @staticmethod
    def _materialize_live_state_h5(b64_token: str, dest_h5_path: str) -> bool:
        """Decode a ``[subhkl-live-state] <b64>`` stdout snapshot into an HDF5 file for live zone rendering."""
        import base64
        import zlib

        import h5py
        import numpy as np
        from scipy.spatial.transform import Rotation

        from subhkl.config import beamlines
        from subhkl.instrument.detector import Detector

        raw_json = zlib.decompress(base64.b64decode(b64_token.strip())).decode("utf-8")
        doc = json.loads(raw_json)
        if not isinstance(doc, dict) or "U" not in doc or "B" not in doc:
            return False
        inst = str(doc.get("instrument") or "CG4D")
        sg = str(doc.get("space_group") or "P 1")
        wl = doc.get("wavelength") or [2.8, 4.5]
        cell_vals = [
            float(x) for x in (doc.get("cell") or [68.7, 69.8, 156.9, 90.0, 90.0, 90.0])
        ]
        tmp_p = dest_h5_path + ".tmp.h5"
        try:
            with h5py.File(tmp_p, "w") as hf:
                hf.attrs["instrument"] = inst
                hf["instrument/name"] = inst
                hf["instrument/wavelength"] = np.asarray(wl[:2], dtype=float)
                hf["sample/cell"] = np.asarray(cell_vals[:6], dtype=float)
                for k_c, v_c in zip(
                    ("a", "b", "c", "alpha", "beta", "gamma"), cell_vals[:6]
                ):
                    hf[f"sample/{k_c}"] = float(v_c)
                hf["sample/space_group"] = sg
                hf["sample/U"] = np.asarray(doc["U"], dtype=float)
                hf["sample/B"] = np.asarray(doc["B"], dtype=float)
                if doc.get("R") is not None:
                    hf["goniometer/R"] = np.asarray(doc["R"], dtype=float)
                banks_map = doc.get("banks")
                if isinstance(banks_map, dict) and banks_map:
                    cal_g = hf.create_group("detector_calibration")
                    for b_id, b_info in banks_map.items():
                        bg = cal_g.create_group(f"bank_{b_id}")
                        bg["center"] = np.asarray(b_info["center"], dtype=float)
                        bg["uhat"] = np.asarray(b_info["uhat"], dtype=float)
                        bg["vhat"] = np.asarray(b_info["vhat"], dtype=float)
                        bg["width"] = float(b_info["width"])
                        bg["height"] = float(b_info["height"])
                elif (
                    "radial_scale" in doc
                    or "axial_scale" in doc
                    or "global_rot_vec" in doc
                    or "global_trans_m" in doc
                ) and inst in beamlines:
                    r_sc = float(doc.get("radial_scale", 0.0))
                    ax_sc = float(doc.get("axial_scale", r_sc))
                    scale_vec = np.array(
                        [1.0 + r_sc, 1.0 + ax_sc, 1.0 + r_sc], dtype=float
                    )
                    r_vec = np.asarray(
                        doc.get("global_rot_vec", [0.0, 0.0, 0.0]), dtype=float
                    )
                    t_vec = np.asarray(
                        doc.get("global_trans_m", [0.0, 0.0, 0.0]), dtype=float
                    )
                    rot_m = Rotation.from_rotvec(r_vec).as_matrix()
                    cal_g = hf.create_group("detector_calibration")
                    for b_k, b_cfg in beamlines[inst].items():
                        try:
                            b_id_int = int(b_k)
                        except (ValueError, TypeError):
                            continue
                        det_b = Detector(dict(b_cfg))
                        mid_b = (
                            det_b.center
                            + 0.5 * det_b.width * det_b.uhat
                            + 0.5 * det_b.height * det_b.vhat
                        )
                        mid_cal = (mid_b * scale_vec) @ rot_m.T + t_vec
                        uh_cal = det_b.uhat @ rot_m.T
                        vh_cal = det_b.vhat @ rot_m.T
                        cen_cal = (
                            mid_cal
                            - 0.5 * det_b.width * uh_cal
                            - 0.5 * det_b.height * vh_cal
                        )
                        bg = cal_g.create_group(f"bank_{b_id_int}")
                        bg["center"] = np.asarray(cen_cal, dtype=float)
                        bg["uhat"] = np.asarray(uh_cal, dtype=float)
                        bg["vhat"] = np.asarray(vh_cal, dtype=float)
                        bg["width"] = float(det_b.width)
                        bg["height"] = float(det_b.height)
                if doc.get("kind") == "stages":
                    rep = hf.create_group("scan_refinement")
                    rep["U"] = np.asarray(doc["U"], dtype=float)
                    rep["B"] = np.asarray(doc["B"], dtype=float)
                    rep["band_wavelength"] = np.asarray(
                        doc.get("band_wavelength") or wl[:2], dtype=float
                    )
                    rep.attrs["stages"] = json.dumps(doc.get("stages") or [])
                    rep.attrs["status"] = "running"
                    rep.attrs["n_zones"] = int(doc.get("n_zones", 120))
                    rep.attrs["d_min"] = float(doc.get("d_min", 1.5))
                else:
                    sg_grp = hf.create_group("search")
                    sg_grp.attrs["instrument"] = inst
                    sg_grp.attrs["d_min"] = float(doc.get("d_min", 1.5))
                    sg_grp.attrs["ladder_stage"] = str(doc.get("stage") or "blind")
                    sg_grp.attrs["ladder_zones"] = int(doc.get("n_zones", 60))
            os.replace(tmp_p, dest_h5_path)
            return True
        finally:
            if os.path.exists(tmp_p):
                with contextlib.suppress(OSError):
                    os.unlink(tmp_p)

    def _auto_discover_local_galaxy(self) -> None:
        """Auto-discover local Planemo / Galaxy server if env vars were not injected."""
        if self.galaxy_url and self.galaxy_url.startswith("https://"):
            http_alt = "http://" + self.galaxy_url[len("https://") :]
            if (
                any(h in self.galaxy_url for h in ("localhost", "127.0.0.1"))
                and not self._is_galaxy_server(self.galaxy_url)
                and self._is_galaxy_server(http_alt)
            ):
                self.galaxy_url = http_alt

        if not self.galaxy_url:
            for candidate in ("http://127.0.0.1:9090", "http://localhost:9090"):
                if self._is_galaxy_server(candidate):
                    self.galaxy_url = candidate
                    break

        if self.galaxy_url and (
            not self.galaxy_api_key or self.galaxy_api_key == "test_key"
        ):
            # Check if Planemo's default master_api_key ('test_key') can mint/fetch a user key
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                users = self._api_request("GET", "/api/users", api_key="test_key")
                if isinstance(users, list) and users:
                    uid = users[0]["id"]
                    ukey = self._api_request(
                        "GET", f"/api/users/{uid}/api_key", api_key="test_key"
                    )
                    if not (isinstance(ukey, str) and ukey):
                        ukey = self._api_request(
                            "POST", f"/api/users/{uid}/api_key", api_key="test_key"
                        )
                    if isinstance(ukey, str) and ukey:
                        self.galaxy_api_key = ukey
                    elif not self.galaxy_api_key:
                        self.galaxy_api_key = "test_key"

        if self.galaxy_url and self.galaxy_api_key and not self.history_id:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                histories = self._api_request("GET", "/api/histories")
                if isinstance(histories, list) and histories:
                    self.history_id = histories[0]["id"]

    @property
    def is_connected(self) -> bool:
        return bool(self.galaxy_url and self.galaxy_api_key)

    def _api_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        raw: bool = False,
        timeout: float | None = None,
    ) -> Any:
        if not self.galaxy_url:
            raise RuntimeError("GALAXY_URL is not configured.")
        key = api_key or self.galaxy_api_key
        sep = "&" if "?" in path else "?"
        url = f"{self.galaxy_url}{path}"
        if key:
            url = f"{url}{sep}key={urllib.parse.quote(key)}"
        headers = {"Accept": "*/*" if raw else "application/json"}
        if key:
            headers["x-api-key"] = key
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req_timeout = timeout if timeout is not None else (180.0 if raw else 30.0)
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=req_timeout) as resp:
            body = resp.read()
            if raw:
                return body
            return json.loads(body.decode("utf-8"))

    @staticmethod
    def _link_or_copy_file(src: str, dst: str, *, allow_hardlink: bool = True) -> None:
        """Atomically stage ``src`` at ``dst``, preferring zero-copy ``os.link`` inside temp/work dirs."""
        abs_s = os.path.abspath(src)
        abs_d = os.path.abspath(dst)
        if abs_s == abs_d:
            return
        if os.path.isfile(abs_d):
            with contextlib.suppress(OSError):
                if os.path.samefile(abs_s, abs_d):
                    return
        parent_d = os.path.dirname(abs_d)
        if parent_d:
            os.makedirs(parent_d, exist_ok=True)
        tmp_dst = f"{abs_d}.stage.{os.getpid()}.{threading.get_ident()}"
        staging_roots = {
            _get_staging_root(),
            os.path.abspath(tempfile.gettempdir()),
        }
        can_link = (
            allow_hardlink
            and (
                any(abs_s.startswith(r + os.sep) for r in staging_roots)
                or os.path.dirname(abs_s) == parent_d
            )
            and os.access(abs_s, os.W_OK)
        )
        try:
            if os.path.lexists(tmp_dst):
                os.remove(tmp_dst)
            if can_link:
                try:
                    os.link(abs_s, tmp_dst)
                    os.replace(tmp_dst, abs_d)
                    return
                except OSError:
                    if os.path.lexists(tmp_dst):
                        with contextlib.suppress(OSError):
                            os.remove(tmp_dst)
            shutil.copy2(abs_s, tmp_dst)
            os.replace(tmp_dst, abs_d)
        finally:
            if os.path.lexists(tmp_dst):
                with contextlib.suppress(OSError):
                    os.remove(tmp_dst)

    def _galaxy_has_datatype(self, ext: str) -> bool:
        """Return True if the connected Galaxy instance registers ``ext`` in ``/api/datatypes``."""
        target = str(ext or "").strip().lower().lstrip(".")
        if not target or not self.galaxy_url:
            return False
        cached = getattr(self, "_cached_galaxy_datatypes", None)
        if isinstance(cached, set):
            return target in cached
        dtypes: set[str] = set()
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            res = self._api_request("GET", "/api/datatypes", timeout=8.0)
            if isinstance(res, list):
                dtypes = {
                    str(x).strip().lower().lstrip(".")
                    for x in res
                    if isinstance(x, (str, int)) and str(x).strip()
                }
                self._cached_galaxy_datatypes = dtypes
        return target in dtypes

    def _is_localhost_galaxy(self) -> bool:
        return bool(
            self.galaxy_url
            and any(h in self.galaxy_url for h in ("127.0.0.1", "localhost"))
        )

    def _should_strip_remote_tof_events(self, dest_path: str) -> bool:
        return bool(
            self.is_connected
            and str(dest_path or "").lower().endswith((".h5", ".hdf5"))
        )

    @staticmethod
    def _read_h5_galaxy_provenance(path: str | None) -> tuple[str, int]:
        if not path or not os.path.isfile(path):
            return "", 0
        if not str(path).lower().endswith((".h5", ".hdf5")):
            return "", 0
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py

            with h5py.File(path, "r") as hf:
                raw_id = hf.attrs.get("galaxy_dataset_id")
                if isinstance(raw_id, (bytes, np.bytes_)):
                    ds_id = raw_id.decode("utf-8", "ignore").strip()
                else:
                    ds_id = str(raw_id or "").strip()
                raw_sz = int(hf.attrs.get("galaxy_file_size") or 0)
                return ds_id, max(0, raw_sz)
        return "", 0

    @staticmethod
    def _write_slim_h5_without_events(
        hf_in: Any,
        dest_part_path: str,
        *,
        ds_id: str = "",
        galaxy_file_size: int = 0,
    ) -> bool:
        """Copy all HDF5 groups/datasets except ``events`` into ``dest_part_path``.

        Preserves LZF-compressed ``images`` chunks verbatim via ``hf_in.copy`` without
        decompression/recompression and stamps ``has_tof_events`` + Galaxy provenance
        attributes on the root group. Returns False if ``hf_in`` lacks ``images`` or ``events``.
        """
        import h5py

        if "images" not in hf_in or "events" not in hf_in:
            return False
        has_ev_tof = "events/tof" in hf_in and "events/image_offsets" in hf_in
        with h5py.File(dest_part_path, "w") as hf_out:
            for attr_k, attr_v in hf_in.attrs.items():
                with contextlib.suppress(TypeError, ValueError, OSError):
                    hf_out.attrs[attr_k] = attr_v
            if has_ev_tof:
                hf_out.attrs["has_tof_events"] = True
            if ds_id:
                hf_out.attrs["galaxy_dataset_id"] = str(ds_id)
            if galaxy_file_size > 0:
                hf_out.attrs["galaxy_file_size"] = int(galaxy_file_size)
            for key in hf_in:
                if key == "events":
                    continue
                hf_in.copy(key, hf_out, name=key)
        return True

    def _load_slim_remote_tof_slab(
        self,
        slim_h5_path: str,
        f0: int,
        f1: int,
        imgs_slab: Any,
    ) -> tuple[Any, Any] | None:
        """Fetch only the active frame's ``[f0:f1]`` ``events/tof`` slice for a slimmed remote HDF5."""
        ds_id, total_sz = self._read_h5_galaxy_provenance(slim_h5_path)
        if not ds_id:
            return None
        try:
            import h5py
            import numpy as np

            # 1) Shared filesystem path if mounted
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                info = self._api_request("GET", f"/api/datasets/{ds_id}", timeout=8.0)
                if isinstance(info, dict):
                    shared_fn = str(info.get("file_name") or "").strip()
                    if shared_fn and os.path.isfile(shared_fn):
                        from subhkl.io.events import EventStore

                        with EventStore(shared_fn) as store:
                            tof_s, off_s, _ = store.read_slab(
                                np.arange(f0, f1, dtype=np.int64), imgs_slab
                            )
                            return tof_s, off_s
                    if total_sz <= 0:
                        total_sz = int(info.get("file_size") or 0)

            # 2) HTTP Range slice read from Galaxy without downloading full multi-GB file
            if not self.galaxy_url or total_sz <= 0:
                return None
            import requests

            path = f"/api/datasets/{ds_id}/display"
            url = f"{self.galaxy_url}{path}"
            headers: dict[str, str] = {"Accept": "*/*", "X-Accel-Buffering": "no"}
            if self.galaxy_api_key:
                url = f"{url}?key={urllib.parse.quote(self.galaxy_api_key)}"
                headers["x-api-key"] = self.galaxy_api_key
            probe_h = dict(headers)
            probe_h["Range"] = (
                f"bytes=0-{min(total_sz - 1, _HTTP_RANGE_BLOCK_BYTES - 1)}"
            )
            resp = requests.get(url, headers=probe_h, stream=True, timeout=(10.0, 30.0))
            if resp.status_code != 206:
                resp.close()
                return None
            first_blk = resp.content
            resp.close()
            rng = _HttpRangeReader(
                url=url,
                headers=headers,
                total_size=total_sz,
                initial_block=first_blk,
            )
            try:
                with h5py.File(rng, "r") as hf_r:
                    if "events/tof" not in hf_r or "events/image_offsets" not in hf_r:
                        return None
                    img_offs = np.asarray(
                        hf_r["events/image_offsets"][()], dtype=np.int64
                    )
                    lo_ev = int(img_offs[f0])
                    hi_ev = int(img_offs[f1])
                    tof_s = np.asarray(
                        hf_r["events/tof"][lo_ev:hi_ev], dtype=np.float32
                    )
                    n_imgs = f1 - f0
                    n_pix = int(imgs_slab[0].size) if n_imgs > 0 else 0
                    off_s = np.empty((n_imgs, n_pix + 1), dtype=np.int64)
                    counts = np.asarray(imgs_slab, dtype=np.int64).reshape(
                        n_imgs, n_pix
                    )
                    np.cumsum(counts, axis=1, out=off_s[:, 1:])
                    bases = img_offs[f0:f1] - lo_ev
                    off_s[:, 0] = bases
                    off_s[:, 1:] += bases[:, None]
                    return tof_s, off_s
            finally:
                rng.close()
        except (_RangeFallbackError, *_NOVA_RECOVERABLE_ERRORS):
            return None

    @staticmethod
    def _check_scratch_space(parent_dir: str, required_bytes: int, label: str) -> None:
        if required_bytes <= 0 or not parent_dir or not os.path.isdir(parent_dir):
            return
        try:
            free_b = int(shutil.disk_usage(parent_dir).free)
        except OSError:
            return
        headroom = 32 * 1024 * 1024
        if free_b > 0 and free_b < required_bytes + headroom:
            raise OSError(
                errno.ENOSPC,
                f"Insufficient scratch space in {parent_dir} "
                f"({free_b / (1024 * 1024):.0f} MB free, "
                f"{required_bytes / (1024 * 1024):.0f} MB needed for '{label}')",
            )

    @staticmethod
    def _set_stream_progress_line(
        job_record: dict[str, Any] | None, marker: str, line_text: str
    ) -> None:
        if not isinstance(job_record, dict):
            return
        cur = str(job_record.get("stdout") or "")
        kept = [ln for ln in cur.splitlines() if marker not in ln]
        if line_text:
            kept.append(line_text)
        job_record["stdout"] = ("\n".join(kept) + "\n") if kept else ""

    def _download_dataset_to_file(
        self,
        ds_id: str,
        dest_path: str,
        *,
        job_record: dict[str, Any] | None = None,
        label: str = "",
        ds_info: dict[str, Any] | None = None,
        timeout: float = 240.0,
        strip_tof_events: bool | None = None,
        resume_part_path: str | None = None,
    ) -> int:
        """Stream a Galaxy dataset directly to ``dest_path`` without buffering hundreds of MB in RAM.

        For large remote HDF5 image stacks carrying multi-GB ``events/tof`` arrays, uses HTTP
        ``Range`` requests + ``h5py`` raw chunk copy to materialize only the 2D ``images`` stack
        and metadata locally (~1-2% of total size), avoiding reverse-proxy buffering timeouts and
        container ``/tmp`` exhaustion while leaving ``events/tof`` intact in Galaxy for GPU jobs.
        """
        disp_lbl = label or os.path.basename(dest_path) or str(ds_id)
        if strip_tof_events is None:
            strip_tof_events = self._should_strip_remote_tof_events(dest_path)
        if isinstance(job_record, dict):
            job_record.pop("_last_slim_remote_bytes", None)
        info_doc = ds_info if isinstance(ds_info, dict) else None
        if info_doc is None and "_api_request" not in self.__dict__:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                fetched = self._api_request(
                    "GET", f"/api/datasets/{ds_id}", timeout=10.0
                )
                if isinstance(fetched, dict):
                    info_doc = fetched

        parent_d = os.path.dirname(os.path.abspath(dest_path))
        if parent_d:
            os.makedirs(parent_d, exist_ok=True)
        part_path = resume_part_path or (
            f"{os.path.abspath(dest_path)}.part.{os.getpid()}.{threading.get_ident()}"
        )
        prog_marker = f"[nova <- galaxy] Syncing '{disp_lbl}'"

        if isinstance(info_doc, dict):
            shared_fn = str(info_doc.get("file_name") or "").strip()
            if (
                shared_fn
                and os.path.isfile(shared_fn)
                and os.path.getsize(shared_fn) > 0
            ):
                shared_sz = os.path.getsize(shared_fn)
                if (
                    strip_tof_events
                    and not self._is_localhost_galaxy()
                    and shared_sz >= _SLIM_REMOTE_H5_MIN_BYTES
                ):
                    slim_tmp = f"{part_path}.slim"
                    try:
                        self._set_stream_progress_line(
                            job_record,
                            prog_marker,
                            f"{prog_marker} — extracting 2D image stack from shared storage ({shared_sz / (1024 * 1024):.0f} MB)...",
                        )
                        import h5py

                        with h5py.File(shared_fn, "r") as hf_sh:
                            if self._write_slim_h5_without_events(
                                hf_sh,
                                slim_tmp,
                                ds_id=ds_id,
                                galaxy_file_size=shared_sz,
                            ):
                                os.replace(slim_tmp, dest_path)
                                if isinstance(job_record, dict):
                                    job_record["_last_slim_remote_bytes"] = shared_sz
                                self._set_stream_progress_line(
                                    job_record, prog_marker, ""
                                )
                                return os.path.getsize(dest_path)
                    except _NOVA_RECOVERABLE_ERRORS:
                        pass
                    finally:
                        if os.path.lexists(slim_tmp):
                            with contextlib.suppress(OSError):
                                os.remove(slim_tmp)
                self._check_scratch_space(parent_d, shared_sz, disp_lbl)
                if shared_sz >= 16 * 1024 * 1024:
                    self._set_stream_progress_line(
                        job_record,
                        prog_marker,
                        f"{prog_marker} — copying {shared_sz / (1024 * 1024):.1f} MB from shared storage...",
                    )
                try:
                    # Copy (never hard-link external Galaxy object-store files) so local HDF5 stamping is safe
                    self._link_or_copy_file(shared_fn, dest_path, allow_hardlink=False)
                finally:
                    self._set_stream_progress_line(job_record, prog_marker, "")
                return os.path.getsize(dest_path)

        # Preserve compatibility when unit tests monkeypatch instance ``_api_request`` directly
        if "_api_request" in self.__dict__:
            raw_bytes = self._api_request(
                "GET",
                f"/api/datasets/{ds_id}/display",
                raw=True,
                timeout=timeout,
            )
            if not raw_bytes:
                return 0
            try:
                with open(part_path, "wb") as fh:
                    fh.write(raw_bytes)
                os.replace(part_path, dest_path)
            finally:
                if os.path.lexists(part_path):
                    with contextlib.suppress(OSError):
                        os.remove(part_path)
            return len(raw_bytes)

        if not self.galaxy_url:
            raise RuntimeError("GALAXY_URL is not configured.")
        key = self.galaxy_api_key
        path = f"/api/datasets/{ds_id}/display"
        sep = "&" if "?" in path else "?"
        url = f"{self.galaxy_url}{path}"
        if key:
            url = f"{url}{sep}key={urllib.parse.quote(key)}"
        headers = {
            "Accept": "*/*",
            "X-Accel-Buffering": "no",
        }
        if key:
            headers["x-api-key"] = key
        expected_bytes = (
            int(info_doc.get("file_size") or 0) if isinstance(info_doc, dict) else 0
        )

        # Emit immediate status BEFORE opening the HTTP stream so slow reverse proxies never appear frozen
        if expected_bytes > 0:
            self._set_stream_progress_line(
                job_record,
                prog_marker,
                f"{prog_marker} — connecting to Galaxy ({expected_bytes / (1024 * 1024):.1f} MB)...",
            )
        else:
            self._set_stream_progress_line(
                job_record,
                prog_marker,
                f"{prog_marker} — connecting to Galaxy...",
            )

        # Fast-path 1: HTTP Range + h5py raw object-copy for large remote HDF5 image+event stacks
        prefetched_stream_resp: Any = None
        if strip_tof_events and (
            expected_bytes <= 0 or expected_bytes >= _SLIM_REMOTE_H5_MIN_BYTES
        ):
            slim_part = f"{part_path}.slim"
            try:
                import h5py
                import requests

                probe_headers = dict(headers)
                probe_end = (
                    min(expected_bytes - 1, _HTTP_RANGE_BLOCK_BYTES - 1)
                    if expected_bytes > 0
                    else (_HTTP_RANGE_BLOCK_BYTES - 1)
                )
                probe_headers["Range"] = f"bytes=0-{probe_end}"
                probe_resp = requests.get(
                    url,
                    headers=probe_headers,
                    stream=True,
                    timeout=(20.0, 60.0),
                )
                if probe_resp.status_code == 206:
                    try:
                        cr_hdr = str(probe_resp.headers.get("Content-Range") or "")
                        m_cr = re.search(r"/([0-9]+)\s*$", cr_hdr)
                        total_sz = (
                            int(m_cr.group(1)) if m_cr else int(expected_bytes or 0)
                        )
                        first_blk = probe_resp.content
                    finally:
                        probe_resp.close()
                    if (
                        total_sz >= _SLIM_REMOTE_H5_MIN_BYTES
                        and len(first_blk) >= 8
                        and first_blk[:8] == b"\x89HDF\r\n\x1a\n"
                    ):
                        expected_bytes = total_sz
                        last_rng_t = time.monotonic()

                        def _on_rng_prog(fetched_b: int, tot_b: int) -> None:
                            nonlocal last_rng_t
                            now_r = time.monotonic()
                            if job_record is None or (now_r - last_rng_t) < 0.5:
                                return
                            last_rng_t = now_r
                            self._set_stream_progress_line(
                                job_record,
                                prog_marker,
                                f"{prog_marker} — {fetched_b / (1024 * 1024):.1f} MB 2D stack fetched via HTTP Range "
                                f"(skipping {tot_b / (1024 * 1024 * 1024):.2f} GB TOF events)...",
                            )

                        self._set_stream_progress_line(
                            job_record,
                            prog_marker,
                            f"{prog_marker} — {len(first_blk) / (1024 * 1024):.1f} MB 2D stack fetched via HTTP Range "
                            f"(skipping {total_sz / (1024 * 1024 * 1024):.2f} GB TOF events)...",
                        )
                        rng_reader = _HttpRangeReader(
                            url=url,
                            headers=headers,
                            total_size=total_sz,
                            initial_block=first_blk,
                            on_progress=_on_rng_prog,
                        )
                        try:
                            with h5py.File(rng_reader, "r") as hf_remote:
                                wrote_slim = self._write_slim_h5_without_events(
                                    hf_remote,
                                    slim_part,
                                    ds_id=ds_id,
                                    galaxy_file_size=total_sz,
                                )
                        finally:
                            rng_reader.close()
                        if wrote_slim and os.path.isfile(slim_part):
                            os.replace(slim_part, dest_path)
                            if isinstance(job_record, dict):
                                job_record["_last_slim_remote_bytes"] = total_sz
                            self._set_stream_progress_line(job_record, prog_marker, "")
                            return os.path.getsize(dest_path)
                elif probe_resp.status_code == 200:
                    # Server ignored Range and started sending the full body; reuse open stream
                    prefetched_stream_resp = probe_resp
                else:
                    probe_resp.close()
            except (_RangeFallbackError, *_NOVA_RECOVERABLE_ERRORS):
                if prefetched_stream_resp is not None:
                    with contextlib.suppress(Exception):
                        prefetched_stream_resp.close()
                    prefetched_stream_resp = None
            finally:
                if os.path.lexists(slim_part):
                    with contextlib.suppress(OSError):
                        os.remove(slim_part)

        # Standard sequential streaming download (supports resuming partial file when resume_part_path is provided)
        chunk_sz = 4 * 1024 * 1024
        resume_offset = 0
        if (
            prefetched_stream_resp is None
            and resume_part_path
            and os.path.isfile(part_path)
        ):
            with contextlib.suppress(OSError):
                resume_offset = os.path.getsize(part_path)
        if expected_bytes > resume_offset:
            self._check_scratch_space(
                parent_d, expected_bytes - resume_offset, disp_lbl
            )
        written = resume_offset
        last_log_t = time.monotonic()
        keep_partial_on_error = bool(resume_part_path)
        try:
            if prefetched_stream_resp is not None:
                try:
                    if expected_bytes <= 0:
                        with contextlib.suppress(ValueError, TypeError):
                            expected_bytes = int(
                                prefetched_stream_resp.headers.get("Content-Length")
                                or 0
                            )
                    if expected_bytes > 0:
                        self._check_scratch_space(parent_d, expected_bytes, disp_lbl)
                    written = 0
                    with open(part_path, "wb") as fh:
                        for chunk in prefetched_stream_resp.iter_content(
                            chunk_size=chunk_sz
                        ):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            written += len(chunk)
                            now_t = time.monotonic()
                            if (
                                job_record is not None
                                and written >= 4 * 1024 * 1024
                                and (now_t - last_log_t) >= 0.8
                            ):
                                last_log_t = now_t
                                mb_done = written / (1024 * 1024)
                                if expected_bytes > 0:
                                    mb_tot = expected_bytes / (1024 * 1024)
                                    pct = min(
                                        99, round(100.0 * written / expected_bytes)
                                    )
                                    msg = f"{prog_marker} — {mb_done:.1f} / {mb_tot:.1f} MB ({pct}%)..."
                                else:
                                    msg = f"{prog_marker} — {mb_done:.1f} MB..."
                                self._set_stream_progress_line(
                                    job_record, prog_marker, msg
                                )
                finally:
                    prefetched_stream_resp.close()
            else:
                req_headers = dict(headers)
                if resume_offset > 0:
                    req_headers["Range"] = f"bytes={resume_offset}-"
                req = urllib.request.Request(url, headers=req_headers, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status_code = int(getattr(resp, "status", 200) or 200)
                    file_mode = (
                        "ab" if (resume_offset > 0 and status_code == 206) else "wb"
                    )
                    if file_mode == "wb":
                        written = 0
                    if expected_bytes <= 0:
                        with contextlib.suppress(ValueError, TypeError):
                            clen = int(resp.headers.get("Content-Length") or 0)
                            if clen > 0:
                                expected_bytes = (
                                    written + clen if file_mode == "ab" else clen
                                )
                    if expected_bytes > written:
                        self._check_scratch_space(
                            parent_d, expected_bytes - written, disp_lbl
                        )
                    if expected_bytes >= 8 * 1024 * 1024:
                        pct0 = (
                            min(99, round(100.0 * written / expected_bytes))
                            if expected_bytes > 0
                            else 0
                        )
                        self._set_stream_progress_line(
                            job_record,
                            prog_marker,
                            f"{prog_marker} — {written / (1024 * 1024):.1f} / {expected_bytes / (1024 * 1024):.1f} MB ({pct0}%)...",
                        )
                    with open(part_path, file_mode) as fh:
                        while True:
                            chunk = resp.read(chunk_sz)
                            if not chunk:
                                break
                            fh.write(chunk)
                            written += len(chunk)
                            now_t = time.monotonic()
                            if (
                                job_record is not None
                                and written >= 8 * 1024 * 1024
                                and (now_t - last_log_t) >= 0.8
                            ):
                                last_log_t = now_t
                                mb_done = written / (1024 * 1024)
                                if expected_bytes > 0:
                                    mb_tot = expected_bytes / (1024 * 1024)
                                    pct = min(
                                        99, round(100.0 * written / expected_bytes)
                                    )
                                    msg = f"{prog_marker} — {mb_done:.1f} / {mb_tot:.1f} MB ({pct}%)..."
                                else:
                                    msg = f"{prog_marker} — {mb_done:.1f} MB..."
                                self._set_stream_progress_line(
                                    job_record, prog_marker, msg
                                )
            if written > 0:
                os.replace(part_path, dest_path)
            self._set_stream_progress_line(job_record, prog_marker, "")
            return written
        except BaseException as exc:
            self._set_stream_progress_line(job_record, prog_marker, "")
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
                keep_partial_on_error = False
            raise
        finally:
            if not keep_partial_on_error and os.path.lexists(part_path):
                with contextlib.suppress(OSError):
                    os.remove(part_path)

    def touch_dataset(self, ds_id: str | None) -> None:
        """Record that a Galaxy dataset ID was just accessed, created, or linked."""
        if not ds_id:
            return
        touch_map: dict[str, float] = getattr(self, "_dataset_touch_times", {})
        touch_map[str(ds_id)] = time.time()
        self._dataset_touch_times = touch_map
        self._cached_history_datasets = None
        self._cached_history_datasets_ts = 0.0

    @staticmethod
    def _normalize_cell_meta(
        cell: Any,
        space_group: Any = "",
        instrument: Any = "",
    ) -> dict[str, Any] | None:
        if not isinstance(cell, (list, tuple)) or len(cell) < 6:
            return None
        try:
            vals = [round(float(x), 5) for x in cell[:6]]
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(v) or v <= 0.0 for v in vals):
            return None
        if any(v >= 180.0 for v in vals[3:6]):
            return None
        sg_str = str(space_group or "").strip()
        inst_str = str(instrument or "").strip()
        res: dict[str, Any] = {
            "cell": vals,
            "space_group": sg_str,
        }
        if inst_str:
            res["instrument"] = inst_str
        return res

    @classmethod
    def _parse_dataset_annotation_meta(
        cls, annotation: Any, tags: Any = None
    ) -> dict[str, Any] | None:
        if isinstance(annotation, dict):
            if "annotation" in annotation or "tags" in annotation:
                if tags is None:
                    tags = annotation.get("tags")
                annotation = annotation.get("annotation")
            else:
                cand = cls._normalize_cell_meta(
                    annotation.get("subhkl_cell") or annotation.get("cell"),
                    annotation.get("space_group"),
                    annotation.get("instrument"),
                )
                if cand:
                    return cand
        if isinstance(annotation, str):
            s = annotation.strip()
            if s.startswith("{") and s.endswith("}"):
                with contextlib.suppress(ValueError, TypeError):
                    doc = json.loads(s)
                    if isinstance(doc, dict):
                        cand = cls._normalize_cell_meta(
                            doc.get("subhkl_cell") or doc.get("cell"),
                            doc.get("space_group"),
                            doc.get("instrument"),
                        )
                        if cand:
                            return cand
        if isinstance(tags, list):
            for t in tags:
                ts = str(t or "").strip()
                if ts.startswith("subhkl_cell:"):
                    parts = ts[len("subhkl_cell:") :].split(":")
                    if len(parts) >= 1:
                        with contextlib.suppress(ValueError, TypeError):
                            c_vals = [
                                float(x) for x in parts[0].split(",") if x.strip()
                            ]
                            sg_tag = (
                                parts[1].replace("_", " ") if len(parts) > 1 else ""
                            )
                            inst_tag = parts[2] if len(parts) > 2 else ""
                            cand = cls._normalize_cell_meta(c_vals, sg_tag, inst_tag)
                            if cand:
                                return cand
        return None

    @classmethod
    def _extract_cell_meta_from_local_file(
        cls, path: str | None
    ) -> dict[str, Any] | None:
        if not path or not os.path.isfile(str(path)):
            return None
        p_str = str(path)
        low = p_str.lower()
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            if low.endswith(".mtz"):
                import gemmi

                mtz = gemmi.read_mtz_file(p_str)
                c_params = list(mtz.cell.parameters)
                sg_hm = mtz.spacegroup.hm if mtz.spacegroup else ""
                return cls._normalize_cell_meta(c_params, sg_hm, "")
            import h5py

            with h5py.File(p_str, "r") as hf:
                raw_cell: list[float] | None = None
                if all(
                    f"sample/{k}" in hf
                    for k in ("a", "b", "c", "alpha", "beta", "gamma")
                ):
                    raw_cell = [
                        float(np.asarray(hf[f"sample/{k}"][()]).flat[0])
                        for k in ("a", "b", "c", "alpha", "beta", "gamma")
                    ]
                elif "sample/B" in hf:
                    from subhkl.viz.rainbow_zones import cell_params_from_B

                    B = np.asarray(hf["sample/B"][()], dtype=np.float64)
                    if B.shape == (3, 3):
                        raw_cell = list(cell_params_from_B(B))
                if not raw_cell:
                    return None
                sg_val = ""
                for sk in ("space_group", "sample/space_group"):
                    if sk in hf:
                        v = hf[sk][()]
                        if hasattr(v, "flat") and v.size > 0:
                            v = v.flat[0]
                        sg_val = (
                            v.decode("utf-8", "replace")
                            if isinstance(v, (bytes, np.bytes_))
                            else str(v)
                        ).strip()
                        if sg_val:
                            break
                if not sg_val and "space_group" in hf.attrs:
                    av = hf.attrs["space_group"]
                    sg_val = (
                        av.decode("utf-8", "replace")
                        if isinstance(av, (bytes, np.bytes_))
                        else str(av)
                    ).strip()
                inst_val = ""
                if "instrument" in hf.attrs:
                    iv = hf.attrs["instrument"]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                elif "instrument/name" in hf:
                    iv = hf["instrument/name"][()]
                    if hasattr(iv, "flat") and iv.size > 0:
                        iv = iv.flat[0]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                elif "instrument" in hf and isinstance(hf["instrument"], h5py.Dataset):
                    iv = hf["instrument"][()]
                    if hasattr(iv, "flat") and iv.size > 0:
                        iv = iv.flat[0]
                    inst_val = (
                        iv.decode("utf-8", "replace")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                return cls._normalize_cell_meta(raw_cell, sg_val, inst_val)
        return None

    def attach_dataset_cell_metadata(
        self,
        ds_id: str | None,
        cell: Any,
        space_group: str | None = "",
        instrument: str | None = "",
        *,
        sync_remote: bool = True,
    ) -> dict[str, Any] | None:
        """Attach unit cell + space group metadata to a Galaxy dataset ID (in-memory + Galaxy HDA annotation)."""
        import re as _re

        if not ds_id:
            return None
        meta = self._normalize_cell_meta(cell, space_group, instrument)
        if not meta:
            return None
        ds_key = str(ds_id).strip()
        cell_map: dict[str, dict[str, Any]] = getattr(self, "_dataset_cell_meta", {})
        prev = cell_map.get(ds_key)
        if prev and not meta.get("space_group") and prev.get("space_group"):
            meta["space_group"] = prev["space_group"]
        if prev and not meta.get("instrument") and prev.get("instrument"):
            meta["instrument"] = prev["instrument"]
        cell_map[ds_key] = meta
        self._dataset_cell_meta = cell_map

        if (
            sync_remote
            and self.is_connected
            and self.history_id
            and _re.match(r"^[0-9a-fA-F]{8,}$", ds_key)
        ):
            synced_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_remote_synced_meta", {}
            )
            if synced_map.get(ds_key) != meta:
                synced_map[ds_key] = dict(meta)
                self._dataset_remote_synced_meta = synced_map
                ann_doc: dict[str, Any] = {
                    "subhkl_cell": meta["cell"],
                    "space_group": meta.get("space_group") or "",
                }
                if meta.get("instrument"):
                    ann_doc["instrument"] = meta["instrument"]
                ann_str = json.dumps(ann_doc, separators=(",", ":"))
                hist_map = getattr(self, "_dataset_history_by_id", {})
                hist_id = str(hist_map.get(ds_key) or self.history_id)

                def _push_ann() -> None:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        self._api_request(
                            "PUT",
                            f"/api/histories/{hist_id}/contents/{ds_key}",
                            payload={"annotation": ann_str},
                            timeout=5.0,
                        )

                threading.Thread(target=_push_ann, daemon=True).start()
        return meta

    def attach_dataset_cell_metadata_from_file(
        self,
        ds_id: str | None,
        local_path: str | None,
        fallback_params: dict[str, Any] | None = None,
        *,
        sync_remote: bool = True,
    ) -> dict[str, Any] | None:
        if not ds_id:
            return None
        meta = self._extract_cell_meta_from_local_file(local_path)
        params = fallback_params or {}
        if not meta and params:
            raw_cell = params.get("cell")
            if not (isinstance(raw_cell, (list, tuple)) and len(raw_cell) >= 6) and all(
                k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ):
                with contextlib.suppress(TypeError, ValueError):
                    raw_cell = [
                        float(params["a"]),
                        float(params["b"]),
                        float(params["c"]),
                        float(params["alpha"]),
                        float(params["beta"]),
                        float(params["gamma"]),
                    ]
            meta = self._normalize_cell_meta(
                raw_cell,
                params.get("space_group"),
                params.get("instrument"),
            )
        elif meta and params:
            if not meta.get("space_group") and params.get("space_group"):
                meta["space_group"] = str(params["space_group"]).strip()
            if not meta.get("instrument") and params.get("instrument"):
                meta["instrument"] = str(params["instrument"]).strip()
        if not meta:
            return None
        return self.attach_dataset_cell_metadata(
            ds_id,
            meta["cell"],
            meta.get("space_group"),
            meta.get("instrument"),
            sync_remote=sync_remote,
        )

    def _local_cached_files_by_ds_id(self) -> dict[str, str]:
        """Return a fast mapping of ds_id -> cached local file path across staging and /tmp."""
        import re as _re

        scan_dirs: list[str] = []
        for cand_d in (_get_staging_root(), tempfile.gettempdir()):
            if cand_d and os.path.isdir(cand_d):
                abs_d = os.path.abspath(cand_d)
                if abs_d not in scan_dirs:
                    scan_dirs.append(abs_d)
        if not scan_dirs:
            return {}
        with contextlib.suppress(OSError):
            mtimes = tuple(os.stat(d).st_mtime_ns for d in scan_dirs)
            cached_pair = getattr(self, "_tmp_gx_index_cache", None)
            if (
                isinstance(cached_pair, tuple)
                and len(cached_pair) == 2
                and cached_pair[0] == mtimes
            ):
                return cached_pair[1]
            mapping: dict[str, str] = {}
            for d_path in scan_dirs:
                for fn in os.listdir(d_path):
                    if not fn.startswith("subhkl_gx_"):
                        continue
                    m = _re.match(r"^subhkl_gx_([0-9a-fA-F]{8,})_", fn)
                    if m:
                        ds_id = m.group(1)
                        if ds_id not in mapping or fn.endswith((".h5", ".mtz")):
                            mapping[ds_id] = os.path.join(d_path, fn)
            self._tmp_gx_index_cache = (mtimes, mapping)
            return mapping
        return {}

    @staticmethod
    def _parse_galaxy_timestamp(ts_str: str | None) -> float:
        if not ts_str:
            return 0.0
        s = str(ts_str).strip()
        if not s:
            return 0.0
        with contextlib.suppress(ValueError, TypeError):
            from datetime import datetime

            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return float(dt.timestamp())
        return 0.0

    def _resolve_subhkl_datastore_history_id(self) -> str | None:
        """Discover an existing 'IMAGINE-X SubHKL' Galaxy history without creating an empty history."""
        cached_hid = getattr(self, "_subhkl_datastore_history_id", None)
        if cached_hid:
            return str(cached_hid)
        if not self.is_connected:
            return None
        now_ts = time.time()
        last_probe = float(getattr(self, "_probed_subhkl_history_ts", 0.0) or 0.0)
        if (now_ts - last_probe) < 30.0:
            return None
        self._probed_subhkl_history_ts = now_ts
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            hists = self._api_request("GET", "/api/histories?keys=id,name,deleted")
            if isinstance(hists, list):
                for h in hists:
                    if not isinstance(h, dict) or h.get("deleted"):
                        continue
                    h_name = str(h.get("name") or "").strip()
                    h_id = str(h.get("id") or "").strip()
                    if h_id and h_name == "IMAGINE-X SubHKL":
                        if (
                            not getattr(self, "_launch_history_id", None)
                            and self.history_id
                        ):
                            self._launch_history_id = self.history_id
                        self._subhkl_datastore_history_id = h_id
                        self.history_id = h_id
                        return h_id
        return None

    def list_history_datasets(
        self, limit: int | None = None, *, use_cached: bool = False
    ) -> list[dict[str, Any]]:
        """Return active datasets across 'IMAGINE-X SubHKL' and launch Galaxy histories, sorted most-recently-touched first."""
        if not self.is_connected:
            return []
        if not self.history_id:
            self._auto_discover_local_galaxy()
        if not getattr(self, "_explicit_history_id", False):
            self._resolve_subhkl_datastore_history_id()
        if not self.history_id and not getattr(
            self, "_subhkl_datastore_history_id", None
        ):
            return []
        cached_list = getattr(self, "_cached_history_datasets", None)
        cached_ts = float(getattr(self, "_cached_history_datasets_ts", 0.0) or 0.0)
        if (
            use_cached
            and isinstance(cached_list, list)
            and (bool(cached_list) or (time.time() - cached_ts) < 15.0)
        ):
            return (
                cached_list[:limit]
                if isinstance(limit, int) and limit > 0
                else list(cached_list)
            )
        try:
            hist_ids: list[str] = []
            for cand_hid in (
                getattr(self, "_subhkl_datastore_history_id", None),
                self.history_id,
                getattr(self, "_launch_history_id", None),
            ):
                s_hid = str(cand_hid or "").strip()
                if s_hid and s_hid not in hist_ids:
                    hist_ids.append(s_hid)
            if not hist_ids:
                return []

            raw_entries: list[tuple[str, dict[str, Any]]] = []
            seen_ds_ids: set[str] = set()
            for hid_q in hist_ids:
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    items_q = self._api_request(
                        "GET",
                        (
                            f"/api/histories/{hid_q}/contents"
                            "?v=dev&q=deleted&qv=false"
                            "&keys=id,name,hid,state,extension,deleted,history_content_type,update_time,create_time,tags,annotation"
                        ),
                    )
                    if isinstance(items_q, list):
                        for it_q in items_q:
                            if not isinstance(it_q, dict):
                                continue
                            did_q = str(it_q.get("id") or "")
                            if did_q and did_q in seen_ds_ids:
                                continue
                            if did_q:
                                seen_ds_ids.add(did_q)
                            raw_entries.append((hid_q, it_q))
            if not raw_entries:
                self._cached_history_datasets = []
                self._cached_history_datasets_ts = time.time()
                return []
            from subhkl.viz.rainbow_zones import infer_dataset_extension

            touch_map: dict[str, float] = getattr(self, "_dataset_touch_times", {})
            cell_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_cell_meta", {}
            )
            probed_local: set[str] = getattr(self, "_probed_local_cell_ds_ids", set())
            local_by_id = self._local_cached_files_by_ds_id()
            synced_map: dict[str, dict[str, Any]] = getattr(
                self, "_dataset_remote_synced_meta", {}
            )
            hist_by_ds: dict[str, str] = getattr(self, "_dataset_history_by_id", {})
            out: list[dict[str, Any]] = []
            for src_hid, it in raw_entries:
                if (
                    not isinstance(it, dict)
                    or it.get("history_content_type") != "dataset"
                    or it.get("deleted")
                ):
                    continue
                ds_id = str(it.get("id") or "")
                if ds_id:
                    hist_by_ds[ds_id] = src_hid
                ut_str = str(it.get("update_time") or it.get("create_time") or "")
                ut_epoch = self._parse_galaxy_timestamp(ut_str)
                touched_at = max(ut_epoch, float(touch_map.get(ds_id, 0.0)))
                ext_norm = infer_dataset_extension(it.get("name"), it.get("extension"))
                cell_meta = self._parse_dataset_annotation_meta(
                    it.get("annotation"), it.get("tags")
                )
                if cell_meta and ds_id:
                    cell_map[ds_id] = cell_meta
                    synced_map[ds_id] = dict(cell_meta)
                elif ds_id in cell_map:
                    cell_meta = cell_map[ds_id]
                elif ds_id and ds_id in local_by_id and ds_id not in probed_local:
                    probed_local.add(ds_id)
                    cell_meta = self.attach_dataset_cell_metadata_from_file(
                        ds_id, local_by_id[ds_id], sync_remote=True
                    )
                entry: dict[str, Any] = {
                    "id": it.get("id"),
                    "hid": it.get("hid"),
                    "name": it.get("name"),
                    "state": it.get("state"),
                    "extension": ext_norm,
                    "history_id": src_hid,
                    "update_time": ut_str,
                    "touched_at": round(touched_at, 3),
                }
                if cell_meta:
                    entry["cell"] = cell_meta["cell"]
                    entry["space_group"] = cell_meta.get("space_group") or ""
                    if cell_meta.get("instrument"):
                        entry["instrument"] = cell_meta["instrument"]
                out.append(entry)
            self._dataset_history_by_id = hist_by_ds
            self._dataset_cell_meta = cell_map
            self._probed_local_cell_ds_ids = probed_local
            self._dataset_remote_synced_meta = synced_map
            out.sort(
                key=lambda d: (
                    float(d.get("touched_at") or 0.0),
                    int(d.get("hid") or 0),
                ),
                reverse=True,
            )
            self._cached_history_datasets = list(out)
            self._cached_history_datasets_ts = time.time()
            if isinstance(limit, int) and limit > 0:
                return out[:limit]
            return out
        except _NOVA_RECOVERABLE_ERRORS:
            return []

    def resolve_dataset_to_local_h5(
        self, ds_id: str, fallback_dir: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a Galaxy dataset ID to a local HDF5 path (via local file_name, cached ds_id, or /display download)."""
        import glob as _glob

        info = self._api_request("GET", f"/api/datasets/{ds_id}")
        if not isinstance(info, dict):
            raise TypeError(f"Dataset {ds_id} not found on {self.galaxy_url}")
        name = str(info.get("name") or f"{ds_id}.h5")
        state = str(info.get("state") or "ok")
        if state != "ok":
            hid = info.get("hid")
            lbl = f"#{hid} '{name}'" if hid else f"'{name}'"
            raise ValueError(
                f"Cannot ingest Galaxy dataset {lbl}: dataset state is '{state}' (must be 'ok')."
            )
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        ptr_map: dict[str, str] = getattr(self, "_shared_pointer_ds_by_path", {})

        def _remember(path: str) -> str:
            from subhkl.io.shared_storage import (
                read_shared_pointer,
                resolve_input_h5_path,
            )

            ptr_doc = read_shared_pointer(path)
            if ptr_doc is not None:
                try:
                    resolved_sh, _ = resolve_input_h5_path(path, must_exist=True)
                except (FileNotFoundError, ValueError) as exc:
                    hid = info.get("hid")
                    lbl = f"#{hid} '{name}'" if hid else f"'{name}'"
                    raise ValueError(
                        f"Cannot ingest Galaxy shared pointer {lbl}: intermediate '{ptr_doc.get('network_path')}' "
                        f"was already cleaned up from SUBHKL_SHARED_DIR ({exc})."
                    ) from exc
                abs_sh = os.path.abspath(resolved_sh)
                ptr_map[abs_sh] = ds_id
                ptr_map[os.path.realpath(resolved_sh)] = ds_id
                self._shared_pointer_ds_by_path = ptr_map
                register_shared_intermediate(
                    abs_sh,
                    network_path=str(ptr_doc.get("network_path") or ""),
                    dataset_id=ds_id,
                    history_id=str(self.history_id or ""),
                    delete_after_use=bool(ptr_doc.get("delete_after_use")),
                )
                path = abs_sh
            known_map[os.path.abspath(path)] = ds_id
            known_map[os.path.realpath(path)] = ds_id
            self._dataset_id_by_path = known_map
            self.touch_dataset(ds_id)
            self.attach_dataset_cell_metadata_from_file(ds_id, path, sync_remote=True)
            return path

        file_name = str(info.get("file_name") or "")
        if file_name and os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
            return _remember(file_name), info

        # 1. Check cached dataset file across staging root and /tmp by exact ds_id first
        cache_roots: list[str] = []
        for cand_root in (_get_staging_root(), tempfile.gettempdir()):
            if cand_root and os.path.isdir(cand_root):
                abs_r = os.path.abspath(cand_root)
                if abs_r not in cache_roots:
                    cache_roots.append(abs_r)
        for c_root in cache_roots:
            cached_matches = sorted(
                _glob.glob(os.path.join(c_root, f"subhkl_gx_{ds_id}_*"))
            )
            for cm in cached_matches:
                if os.path.isfile(cm) and os.path.getsize(cm) > 0:
                    return _remember(cm), info

        # 2. If dataset name includes a relative folder (e.g. 'betalac-7/merged.h5'), check /home/xvg/subhkl-examples/
        if "/" in name:
            if os.path.isfile(name) and os.path.getsize(name) > 0:
                return _remember(name), info
            ex_cand = os.path.join("/home/xvg/subhkl-examples", name.lstrip("/"))
            if os.path.isfile(ex_cand) and os.path.getsize(ex_cand) > 0:
                return _remember(ex_cand), info

        # 3. Only fall back to fallback_dir if file_size or stamped Galaxy provenance matches
        # (prevents generic 'scan.h5'/'merged.h5' collisions while matching slim 2D caches)
        expected_size = int(info.get("file_size") or 0)
        if fallback_dir and os.path.isdir(fallback_dir):
            cand = os.path.join(fallback_dir, os.path.basename(name))
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                cand_sz = os.path.getsize(cand)
                prov_id, prov_sz = self._read_h5_galaxy_provenance(cand)
                if (
                    prov_id == ds_id
                    or (
                        expected_size > 0
                        and (cand_sz == expected_size or prov_sz == expected_size)
                    )
                    or (expected_size == 0 and not self.is_connected)
                ):
                    return _remember(cand), info

        safe_name = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in os.path.basename(name)
        )
        if not safe_name.endswith((".h5", ".hdf5", ".mtz", ".json", ".nxs")):
            safe_name += ".h5"
        tmp_dest = os.path.join(_get_staging_root(), f"subhkl_gx_{ds_id}_{safe_name}")
        register_temp_artifact(tmp_dest)
        if os.path.isfile(tmp_dest) and os.path.getsize(tmp_dest) > 0:
            return _remember(tmp_dest), info
        n_dl = self._download_dataset_to_file(
            ds_id,
            tmp_dest,
            label=name,
            ds_info=info,
            timeout=240.0,
        )
        if n_dl <= 0 or not os.path.isfile(tmp_dest):
            raise OSError(f"Downloaded dataset {ds_id} ('{name}') was empty.")
        return _remember(tmp_dest), info

    @staticmethod
    def _classify_artifact_stage(filename: str) -> str | None:
        """Map an artifact filename to one of ('reduce', 'solve', 'integrate', 'merge')."""
        low = os.path.basename(filename).lower()
        if not low or low.startswith("."):
            return None
        if any(
            low == k or low.endswith(k)
            for k in (
                "merged.mtz",
                "predictions.mtz",
                "xval.mtz",
                "careless_metrics.json",
                "merge_table.csv",
                "careless_0.mtz",
                "careless_predictions_0.mtz",
                "careless_xval_0.mtz",
            )
        ):
            return "merge"
        if any(
            low == k or low.endswith(k)
            for k in ("predictor.h5", "integrated.h5", "reflections.mtz")
        ):
            return "integrate"
        if "solve" in low and low.endswith(".h5"):
            return "solve"
        if any(
            k in low
            for k in (
                "merged.h5",
                "_merged.h5",
                ".reduce.h5",
                ".nxs.h5",
                "scan_master.h5",
            )
        ):
            return "reduce"
        return None

    def _active_work_dir(self, current_h5: str | None = None) -> str | None:
        cand = current_h5
        if not cand and isinstance(self.active_job, dict):
            cand = self.active_job.get("output_h5_path")
        if cand and os.path.isfile(str(cand)):
            wd, _ = self._resolve_work_dir_and_merged(str(cand))
            if os.path.isdir(wd):
                return wd
            parent = os.path.dirname(os.path.abspath(str(cand)))
            if os.path.isdir(parent):
                return parent
        return None

    STAGE_ORDER: ClassVar[tuple[str, ...]] = (
        "reduce",
        "solve",
        "integrate",
        "merge",
    )

    @classmethod
    def _stage_for_kind_or_stage(cls, val: str | None) -> str | None:
        k = str(val or "").strip().lower()
        if k in cls.STAGE_ORDER:
            return k
        if k in ("merged", "raw_nexus"):
            return "reduce"
        if k in ("solve", "finder"):
            return "solve"
        if k in ("integrator", "predictor"):
            return "integrate"
        if k == "merged_careless":
            return "merge"
        return None

    def invalidate_subsequent_stages(
        self,
        current_stage_or_kind: str | None,
        work_dir: str | None = None,
    ) -> None:
        """Invalidate artifacts and cached state for pipeline stages after ``current_stage_or_kind``."""
        mapped = self._stage_for_kind_or_stage(current_stage_or_kind)
        self._max_valid_stage = mapped
        if not mapped or mapped == "merge":
            return

        wd = work_dir or self._active_work_dir()
        if not wd or not os.path.isdir(wd):
            return

        is_tmp_wd = any(
            os.path.abspath(wd).startswith(r)
            for r in (_get_staging_root(), os.path.abspath(tempfile.gettempdir()))
        )
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        sig_map: dict[str, tuple[int, int, str]] = getattr(
            self, "_dataset_sig_by_path", {}
        )

        to_remove: list[str] = [
            "careless_metrics.json",
            "merge_table.html",
            "merge_table.csv",
            "merged.mtz",
            "predictions.mtz",
            "xval.mtz",
            "careless_0.mtz",
            "careless_predictions_0.mtz",
            "careless_xval_0.mtz",
        ]
        if is_tmp_wd and mapped in ("reduce", "solve"):
            to_remove.extend(
                [
                    "predictor.h5",
                    "peak_predictor.h5",
                    "integrated.h5",
                    "integrator.h5",
                    "reflections.mtz",
                ]
            )
        if is_tmp_wd and mapped == "solve":
            to_remove.extend(
                [
                    "solve.blind.h5",
                    "solve.stages.h5",
                ]
            )
        if is_tmp_wd and mapped == "reduce":
            to_remove.extend(
                [
                    "solve.h5",
                    "peaks.h5",
                    "indexer.h5",
                    "solve.blind.h5",
                    "solve.stages.h5",
                ]
            )

        for fn in to_remove:
            fp = os.path.join(wd, fn)
            for p_key in (os.path.abspath(fp), os.path.realpath(fp)):
                known_map.pop(p_key, None)
                sig_map.pop(p_key, None)
            if os.path.isfile(fp):
                with contextlib.suppress(OSError):
                    os.remove(fp)
        stage_own_files = {
            "reduce": ("merged.h5",),
            "solve": ("solve.h5",),
            "integrate": ("integrated.h5", "predictor.h5", "reflections.mtz"),
        }
        for own_fn in stage_own_files.get(mapped, ()):
            own_fp = os.path.join(wd, own_fn)
            for p_key in (os.path.abspath(own_fp), os.path.realpath(own_fp)):
                known_map.pop(p_key, None)
                sig_map.pop(p_key, None)

    def list_step_artifacts(
        self,
        current_h5: str | None = None,
        history_datasets: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return downloadable artifacts grouped by pipeline stage ('reduce', 'solve', 'integrate', 'merge')."""
        stages: dict[str, dict[str, dict[str, Any]]] = {
            "reduce": {},
            "solve": {},
            "integrate": {},
            "merge": {},
        }
        if history_datasets is None:
            history_datasets = self.list_history_datasets()

        for ds in history_datasets or []:
            if ds.get("state") and ds.get("state") != "ok":
                continue
            raw_name = str(ds.get("name") or "")
            base_name = os.path.basename(raw_name)
            st = self._classify_artifact_stage(base_name)
            if not st or base_name in stages[st]:
                continue
            stages[st][base_name] = {
                "name": base_name,
                "stage_id": st,
                "dataset_id": str(ds.get("id") or ""),
                "hid": ds.get("hid"),
                "source": "NDIP",
            }

        work_dir = self._active_work_dir(current_h5)
        if work_dir and os.path.isdir(work_dir):
            self._ensure_merge_table_artifacts(work_dir, current_h5=current_h5)
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                for fn in sorted(os.listdir(work_dir)):
                    fp = os.path.join(work_dir, fn)
                    if not os.path.isfile(fp) or os.path.getsize(fp) <= 0:
                        continue
                    st = self._classify_artifact_stage(fn)
                    if not st:
                        continue
                    entry = stages[st].get(
                        fn, {"name": fn, "stage_id": st, "source": "LOCAL"}
                    )
                    entry["size_bytes"] = int(os.path.getsize(fp))
                    entry["has_local"] = True
                    stages[st][fn] = entry
                    if fn == "integrated.h5" and os.path.isfile(
                        os.path.join(work_dir, "careless_metrics.json")
                    ):
                        m_entry = dict(entry)
                        m_entry["stage_id"] = "merge"
                        stages["merge"][fn] = m_entry

        max_stage = getattr(self, "_max_valid_stage", None)
        if not max_stage and current_h5:
            max_stage = self._classify_artifact_stage(os.path.basename(str(current_h5)))
            if max_stage == "integrate" and (
                (
                    work_dir
                    and os.path.isfile(os.path.join(work_dir, "careless_metrics.json"))
                )
                or "merge_table.csv" in stages["merge"]
            ):
                max_stage = "merge"
        if max_stage in self.STAGE_ORDER:
            max_idx = self.STAGE_ORDER.index(max_stage)
            for st_k in self.STAGE_ORDER[max_idx + 1 :]:
                stages[st_k] = {}

        return {k: list(v.values()) for k, v in stages.items()}

    def _ensure_merge_table_artifacts(
        self,
        work_dir: str | None,
        *,
        current_h5: str | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """Materialize ``merge_table.csv`` and ``merge_table.html`` when merge metrics exist."""
        from subhkl.viz.rainbow_zones import (
            render_merge_table_csv,
            render_merge_table_html,
        )

        target_dir = work_dir if (work_dir and os.path.isdir(work_dir)) else None
        csv_existing = (
            os.path.join(target_dir, "merge_table.csv") if target_dir else None
        )
        html_existing = (
            os.path.join(target_dir, "merge_table.html") if target_dir else None
        )
        if (
            not force
            and csv_existing
            and html_existing
            and os.path.isfile(csv_existing)
            and os.path.getsize(csv_existing) > 0
            and os.path.isfile(html_existing)
            and os.path.getsize(html_existing) > 0
        ):
            return {
                "merge_table.csv": csv_existing,
                "merge_table.html": html_existing,
            }

        metrics_doc: dict[str, Any] | None = None
        if target_dir:
            m_fp = os.path.join(target_dir, "careless_metrics.json")
            if os.path.isfile(m_fp):
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    with open(m_fp, "r", encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict) and (
                        loaded.get("resolution_bins")
                        or loaded.get("cc_half_bins")
                        or loaded.get("n_merged_asu")
                    ):
                        metrics_doc = loaded

        h5_cand = current_h5
        if (not h5_cand or not os.path.isfile(str(h5_cand))) and target_dir:
            int_cand = os.path.join(target_dir, "integrated.h5")
            if os.path.isfile(int_cand):
                h5_cand = int_cand

        sg_h5: str | None = None
        cell_h5: list[float] | None = None
        if h5_cand and os.path.isfile(str(h5_cand)):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py as _h5py

                with _h5py.File(str(h5_cand), "r") as hf:
                    if metrics_doc is None and "careless/metrics_json" in hf:
                        raw_b = hf["careless/metrics_json"][()]
                        if isinstance(raw_b, (bytes, bytearray)):
                            raw_s = raw_b.decode("utf-8", errors="replace")
                        elif hasattr(raw_b, "tobytes"):
                            raw_s = raw_b.tobytes().decode("utf-8", errors="replace")
                        else:
                            raw_s = str(raw_b)
                        loaded_h5 = json.loads(raw_s)
                        if isinstance(loaded_h5, dict) and (
                            loaded_h5.get("resolution_bins")
                            or loaded_h5.get("cc_half_bins")
                            or loaded_h5.get("n_merged_asu")
                        ):
                            metrics_doc = loaded_h5
                    if "sample" in hf:
                        sgrp = hf["sample"]
                        if "space_group" in sgrp:
                            sg_raw = sgrp["space_group"][()]
                            sg_h5 = (
                                sg_raw.decode("utf-8", errors="replace")
                                if isinstance(sg_raw, (bytes, bytearray))
                                else str(sg_raw)
                            ).strip()
                        if all(
                            k in sgrp for k in ("a", "b", "c", "alpha", "beta", "gamma")
                        ):
                            cell_h5 = [
                                float(sgrp[k][()])
                                for k in ("a", "b", "c", "alpha", "beta", "gamma")
                            ]

        if not isinstance(metrics_doc, dict):
            return {}

        csv_body = render_merge_table_csv(metrics_doc, space_group=sg_h5, cell=cell_h5)
        html_body = render_merge_table_html(
            metrics_doc, space_group=sg_h5, cell=cell_h5
        )
        out_dirs: list[str] = []
        if target_dir:
            out_dirs.append(target_dir)
        fallback_tmp = os.path.join(tempfile.gettempdir(), "subhkl_merge_tables")
        out_dirs.append(fallback_tmp)

        for odir in out_dirs:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                os.makedirs(odir, exist_ok=True)
                if os.path.abspath(odir) == os.path.abspath(fallback_tmp):
                    register_temp_artifact(odir)
                p_csv = os.path.join(odir, "merge_table.csv")
                p_html = os.path.join(odir, "merge_table.html")
                if force or not os.path.isfile(p_csv) or os.path.getsize(p_csv) == 0:
                    with open(p_csv, "w", encoding="utf-8") as fh_c:
                        fh_c.write(csv_body)
                if force or not os.path.isfile(p_html) or os.path.getsize(p_html) == 0:
                    with open(p_html, "w", encoding="utf-8") as fh_h:
                        fh_h.write(html_body)
                return {
                    "merge_table.csv": p_csv,
                    "merge_table.html": p_html,
                }
        return {}

    def resolve_artifact_for_download(
        self,
        *,
        dataset_id: str = "",
        name: str = "",
        current_h5: str | None = None,
    ) -> tuple[str, str]:
        """Resolve an artifact by dataset_id or filename to `(local_file_path, download_filename)`."""
        clean_name = os.path.basename(name.strip()) if name else ""
        work_dir = self._active_work_dir(current_h5)
        if clean_name and work_dir and os.path.isdir(work_dir):
            cand = os.path.join(work_dir, clean_name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand, clean_name

        if clean_name.lower() in ("merge_table.csv", "merge_table.html"):
            gen_map = self._ensure_merge_table_artifacts(
                work_dir, current_h5=current_h5
            )
            gen_fp = gen_map.get(clean_name.lower())
            if gen_fp and os.path.isfile(gen_fp) and os.path.getsize(gen_fp) > 0:
                return gen_fp, clean_name

        ds_id = dataset_id.strip()
        if not ds_id and clean_name:
            for ds in self.list_history_datasets():
                if (
                    ds.get("state") == "ok"
                    and os.path.basename(str(ds.get("name") or "")) == clean_name
                ):
                    ds_id = str(ds.get("id") or "")
                    break

        if ds_id:
            local_p, info = self.resolve_dataset_to_local_h5(
                ds_id, fallback_dir=work_dir
            )
            dl_name = clean_name or os.path.basename(
                str(info.get("name") or os.path.basename(local_p))
            )
            return local_p, dl_name

        raise FileNotFoundError(
            f"Artifact '{clean_name or dataset_id}' not found in work_dir or NDIP Galaxy history."
        )

    def _work_dir_for_reduce(
        self, raw_nxs: list[str], fallback_dir: str
    ) -> tuple[str, str]:
        """Return an isolated per-dataset work directory and descriptive merged filename for raw NeXus reduction."""
        if not raw_nxs:
            return fallback_dir, "merged.h5"
        first_stem = os.path.basename(raw_nxs[0]).split(".")[0]
        last_stem = os.path.basename(raw_nxs[-1]).split(".")[0]
        tag = first_stem if len(raw_nxs) == 1 else f"{first_stem}_{last_stem}"
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        work_d = os.path.join(_get_staging_root(), f"subhkl_work_{safe_tag}")
        os.makedirs(work_d, mode=0o777, exist_ok=True)
        _chmod_world_rw(work_d, is_dir=True)
        register_temp_artifact(work_d)
        return work_d, f"{safe_tag}_merged.h5"

    def _resolve_work_dir_and_merged(self, merged_path: str) -> tuple[str, str]:
        """Ensure a merged HDF5 file inside staging/TMPDIR, shared storage, or Pulsar inputs uses a dedicated subhkl_work_<tag> directory."""
        import glob as _glob
        import re as _re
        import shutil as _shutil

        from subhkl.io.shared_storage import (
            is_safe_shared_intermediate,
            read_shared_pointer,
            resolve_input_h5_path,
        )

        raw_m = str(merged_path)
        if read_shared_pointer(raw_m) is not None:
            with contextlib.suppress(FileNotFoundError, ValueError):
                raw_m, _ = resolve_input_h5_path(raw_m, must_exist=True)
        abs_m = os.path.abspath(raw_m)
        real_m = os.path.abspath(os.path.realpath(abs_m))
        parent_d = os.path.dirname(abs_m)
        orig_bn = os.path.basename(abs_m)
        tmp_root = _get_staging_root()
        tmp_roots = {tmp_root, os.path.abspath(tempfile.gettempdir())}
        is_staging_dat = _is_readonly_galaxy_dat(abs_m)
        is_shared_interm = is_safe_shared_intermediate(real_m) or orig_bn.startswith(
            "subhkl_shared_"
        )
        if parent_d not in tmp_roots and not is_staging_dat and not is_shared_interm:
            return parent_d, abs_m

        m_gx = _re.match(r"^subhkl_gx_([0-9a-fA-F]+)_(.+)$", orig_bn)
        if m_gx and parent_d in tmp_roots:
            register_temp_artifact(abs_m)
        gx_ds_id = m_gx.group(1) if m_gx else None
        clean_bn = m_gx.group(2) if m_gx else orig_bn
        m_sh = _re.match(r"^subhkl_shared_[0-9a-fA-F]{8,}_(.+)$", clean_bn)
        if m_sh:
            clean_bn = m_sh.group(1)

        tag = ""
        is_images_h5 = False
        if os.path.isfile(abs_m):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(abs_m, "r") as hf:
                    is_images_h5 = "images" in hf
                    for ds_key in ("source_files", "files"):
                        if ds_key not in hf:
                            continue
                        raw_f = [
                            (
                                x.decode("utf-8", "ignore")
                                if isinstance(x, bytes)
                                else str(x)
                            )
                            for x in list(hf[ds_key][()])
                        ]
                        raw_f = [x for x in raw_f if x]
                        if raw_f:
                            s0 = os.path.basename(raw_f[0]).split(".")[0]
                            s1 = os.path.basename(raw_f[-1]).split(".")[0]
                            tag = s0 if s0 == s1 else f"{s0}_{s1}"
                            break
                    if not tag:
                        raw_src_attr = hf.attrs.get("source")
                        if raw_src_attr is not None:
                            src_str = (
                                raw_src_attr.decode("utf-8", "ignore")
                                if isinstance(raw_src_attr, bytes)
                                else str(raw_src_attr)
                            ).strip()
                            m_wd = _re.search(r"subhkl_work_([A-Za-z0-9._-]+)", src_str)
                            if m_wd:
                                tag = m_wd.group(1)
                            else:
                                src_bn = _re.sub(
                                    r"([._](?:merged|solve|integrated|integrator|predictor))?\.(h5|hdf5|dat)$",
                                    "",
                                    os.path.basename(src_str),
                                    flags=_re.IGNORECASE,
                                )
                                if src_bn and src_bn.lower() not in (
                                    "merged",
                                    "solve",
                                    "integrated",
                                    "predictor",
                                    "dataset",
                                ):
                                    tag = src_bn
        if not tag:
            bn = _re.sub(
                r"([._](?:merged|solve|integrated|integrator|predictor))?\.(h5|hdf5|dat)$",
                "",
                clean_bn,
                flags=_re.IGNORECASE,
            )
            tag = bn or "dataset"
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        work_d = os.path.join(tmp_root, f"subhkl_work_{safe_tag}")
        os.makedirs(work_d, mode=0o777, exist_ok=True)
        _chmod_world_rw(work_d, is_dir=True)
        register_temp_artifact(work_d)
        dest_name = (
            clean_bn
            if clean_bn.lower().endswith("_merged.h5")
            else f"{safe_tag}_merged.h5"
        )
        dest_m = os.path.join(work_d, dest_name)
        legacy_m = os.path.join(work_d, "merged.h5")
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})

        def _has_h5_images(path_cand: str) -> bool:
            if not os.path.isfile(path_cand):
                return False
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(path_cand, "r") as hf_c:
                    return "images" in hf_c
            return False

        if os.path.isfile(abs_m) and is_images_h5:
            if is_shared_interm:
                _symlink_shared_into_work_dir(real_m, dest_m)
                if dest_m != legacy_m:
                    _symlink_shared_into_work_dir(real_m, legacy_m)
                register_shared_intermediate(real_m, work_dir=work_d)
            else:
                src_sz = os.path.getsize(abs_m)
                if (
                    not os.path.isfile(dest_m)
                    or os.path.getsize(dest_m) != src_sz
                    or not _has_h5_images(dest_m)
                ):
                    try:
                        self._link_or_copy_file(abs_m, dest_m)
                    except OSError:
                        return work_d, abs_m
                if dest_m != legacy_m and (
                    not os.path.isfile(legacy_m)
                    or os.path.getsize(legacy_m) != src_sz
                    or not _has_h5_images(legacy_m)
                ):
                    with contextlib.suppress(OSError):
                        self._link_or_copy_file(abs_m, legacy_m)
            final_m = dest_m if os.path.isfile(dest_m) else abs_m
            mapped_id = (
                known_map.get(abs_m)
                or known_map.get(os.path.realpath(abs_m))
                or gx_ds_id
            )
            if not mapped_id:
                for gx_f in _glob.glob(
                    os.path.join(tmp_root, f"subhkl_gx_*_{safe_tag}_merged.h5")
                ):
                    if not _has_h5_images(gx_f):
                        continue
                    m_cand = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_", os.path.basename(gx_f)
                    )
                    if m_cand:
                        mapped_id = m_cand.group(1)
                        break
            if mapped_id:
                for p_alias in (abs_m, final_m, legacy_m):
                    known_map[os.path.abspath(p_alias)] = mapped_id
                    known_map[os.path.realpath(p_alias)] = mapped_id
                self._dataset_id_by_path = known_map
            return work_d, final_m

        # Non-image HDF5 (e.g. solve.h5, integrated.h5, predictor.h5):
        # Stage canonical stage copy inside work_d and never overwrite <tag>_merged.h5!
        if os.path.isfile(abs_m):
            low_c = clean_bn.lower()
            stage_copy = None
            if "solve" in low_c and low_c.endswith((".h5", ".hdf5")):
                stage_copy = os.path.join(work_d, "solve.h5")
            elif "integrat" in low_c and low_c.endswith((".h5", ".hdf5")):
                stage_copy = os.path.join(work_d, "integrated.h5")
            elif "predict" in low_c and low_c.endswith((".h5", ".hdf5")):
                stage_copy = os.path.join(work_d, "predictor.h5")
            if (
                stage_copy
                and os.path.abspath(stage_copy) != abs_m
                and (
                    not os.path.isfile(stage_copy)
                    or os.path.getsize(stage_copy) != os.path.getsize(abs_m)
                )
            ):
                with contextlib.suppress(OSError):
                    _shutil.copy2(abs_m, stage_copy)
                    _chmod_world_rw(stage_copy, is_dir=False)
            if gx_ds_id:
                known_map[os.path.abspath(abs_m)] = gx_ds_id
                known_map[os.path.realpath(abs_m)] = gx_ds_id
                if stage_copy:
                    known_map[os.path.abspath(stage_copy)] = gx_ds_id
                    known_map[os.path.realpath(stage_copy)] = gx_ds_id
                self._dataset_id_by_path = known_map

        # Self-heal dest_m / legacy_m if missing or previously overwritten by a non-image HDF5
        if not _has_h5_images(dest_m):
            for bad_p in (dest_m, legacy_m):
                if (os.path.islink(bad_p) and not os.path.exists(bad_p)) or (
                    os.path.isfile(bad_p) and not _has_h5_images(bad_p)
                ):
                    with contextlib.suppress(OSError):
                        os.unlink(bad_p)
                known_map.pop(os.path.abspath(bad_p), None)
                known_map.pop(os.path.realpath(bad_p), None)
            for gx_f in sorted(
                _glob.glob(os.path.join(tmp_root, f"subhkl_gx_*_{safe_tag}_merged.h5")),
                key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0.0,
                reverse=True,
            ):
                if _has_h5_images(gx_f):
                    with contextlib.suppress(OSError):
                        _shutil.copy2(gx_f, dest_m)
                        if dest_m != legacy_m:
                            _shutil.copy2(gx_f, legacy_m)
                    m_cand = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_", os.path.basename(gx_f)
                    )
                    if m_cand:
                        cid = m_cand.group(1)
                        for p_alias in (gx_f, dest_m, legacy_m):
                            known_map[os.path.abspath(p_alias)] = cid
                            known_map[os.path.realpath(p_alias)] = cid
                        self._dataset_id_by_path = known_map
                    break

        final_m = dest_m if _has_h5_images(dest_m) else abs_m
        return work_d, final_m

    def _discover_or_fetch_galaxy_images_h5(
        self, peaks_h5_path: str | None
    ) -> str | None:
        """Locate the companion merged scan HDF5 (containing ``images``) on disk or fetch it from Galaxy history."""
        if not peaks_h5_path or not os.path.isfile(str(peaks_h5_path)):
            return None
        from subhkl.viz.rainbow_zones import _discover_images_h5

        # First allow _resolve_work_dir_and_merged to self-heal any local /tmp/subhkl_gx_*_<tag>_merged.h5
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            self._resolve_work_dir_and_merged(str(peaks_h5_path))

        disc = _discover_images_h5(str(peaks_h5_path))
        if disc and os.path.isfile(str(disc)):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(str(disc), "r") as hf_d:
                    if "images" in hf_d:
                        _, resolved_m = self._resolve_work_dir_and_merged(str(disc))
                        return resolved_m

        # In local-staging mode, if the raw NeXus recipe resolves locally on /SNS or /HFIR,
        # avoid blocking /api/payload to download a multi-GB merged.h5 from Galaxy; let
        # _maybe_trigger_hydrate_reduce regenerate frames asynchronously on local NVMe.
        if self._is_local_staging_mode():
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                recipe_local = self._extract_reduce_recipe_from_derived_h5(
                    str(peaks_h5_path)
                )
                if recipe_local and recipe_local.get("nexus_files"):
                    return None

        # Not found locally — inspect `source_files` / `files` in peaks_h5_path and look up <tag>_merged.h5 in Galaxy history
        tag = ""
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py

            with h5py.File(str(peaks_h5_path), "r") as hf:
                for ds_key in ("source_files", "files"):
                    if ds_key not in hf:
                        continue
                    raw_f = [
                        (
                            x.decode("utf-8", "ignore")
                            if isinstance(x, bytes)
                            else str(x)
                        )
                        for x in list(hf[ds_key][()])
                    ]
                    raw_f = [x for x in raw_f if x]
                    if raw_f:
                        s0 = os.path.basename(raw_f[0]).split(".")[0]
                        s1 = os.path.basename(raw_f[-1]).split(".")[0]
                        tag = s0 if s0 == s1 else f"{s0}_{s1}"
                        break
        if not tag or not self.is_connected:
            return None
        safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
        target_names = {
            f"{safe_tag}_merged.h5".lower(),
            f"{tag}_merged.h5".lower(),
            f"{safe_tag}_scan.ref.json".lower(),
            f"{tag}_scan.ref.json".lower(),
            f"{safe_tag}_merged.ref.json".lower(),
        }
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py

            hist_items = self.list_history_datasets(limit=200)
            for ds in hist_items:
                if ds.get("state") != "ok":
                    continue
                dname = str(ds.get("name") or "").strip().lower()
                if dname in target_names or (
                    safe_tag.lower() in dname
                    and dname.endswith(
                        ("_merged.h5", "_scan.ref.json", "_merged.ref.json")
                    )
                ):
                    ds_id = str(ds.get("id") or "").strip()
                    if ds_id:
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            local_m, _ = self.resolve_dataset_to_local_h5(ds_id)
                            _, final_m = self._resolve_work_dir_and_merged(local_m)
                            if final_m and os.path.isfile(final_m):
                                with h5py.File(final_m, "r") as hf_m:
                                    if "images" in hf_m:
                                        return final_m
        return None

    def stage_derived_h5_in_work_dir(
        self,
        local_h5: str,
        *,
        name_hint: str = "",
    ) -> tuple[str, str]:
        """Stage a derived HDF5 (solve/integrated/predictor) into its canonical ``subhkl_work_<tag>`` directory."""
        import shutil as _shutil

        abs_src = os.path.abspath(local_h5)
        work_d, _ = self._resolve_work_dir_and_merged(abs_src)
        hint_low = f"{name_hint} {os.path.basename(abs_src)}".lower()
        canon_name = ""
        if "integrat" in hint_low:
            canon_name = "integrated.h5"
        elif "solve" in hint_low:
            canon_name = "solve.h5"
        elif "predict" in hint_low:
            canon_name = "predictor.h5"
        elif os.path.isfile(abs_src):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(abs_src, "r") as hf_s:
                    if "peaks/intensity" in hf_s:
                        canon_name = "integrated.h5"
                    elif "ladder" in hf_s or "zone_axes" in hf_s or "R" in hf_s:
                        canon_name = "solve.h5"
                    elif "peaks/xyz" in hf_s or "peaks/hkl" in hf_s:
                        canon_name = "predictor.h5"
        if not canon_name or not os.path.isfile(abs_src):
            return work_d, abs_src
        dest_p = os.path.join(work_d, canon_name)
        same_f = os.path.abspath(dest_p) == abs_src
        if not same_f and os.path.exists(dest_p):
            with contextlib.suppress(OSError):
                same_f = os.path.samefile(dest_p, abs_src)
        if not same_f:
            with contextlib.suppress(OSError):
                _shutil.copy2(abs_src, dest_p)
                _chmod_world_rw(dest_p, is_dir=False)
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        mapped_id = known_map.get(abs_src) or known_map.get(os.path.realpath(abs_src))
        if mapped_id and os.path.isfile(dest_p):
            known_map[os.path.abspath(dest_p)] = mapped_id
            known_map[os.path.realpath(dest_p)] = mapped_id
            self._dataset_id_by_path = known_map
        return work_d, (dest_p if os.path.isfile(dest_p) else abs_src)

    def _discover_or_fetch_galaxy_predictor_h5(
        self, peaks_h5_path: str | os.PathLike, *, target_hid: int | None = None
    ) -> str | None:
        """Ensure companion `predictor.h5` from `subhkl peak-predictor` is cached locally when loading `integrated.h5`."""
        from subhkl.viz.rainbow_zones import _discover_predictor_h5

        existing = _discover_predictor_h5(peaks_h5_path)
        if existing and os.path.isfile(existing):
            return existing
        if not self.is_connected:
            return None

        is_integrated = False
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py

            with h5py.File(str(peaks_h5_path), "r") as hf:
                is_integrated = bool(
                    "peaks" in hf
                    and isinstance(hf["peaks"], h5py.Group)
                    and ("var_u" in hf["peaks"] or "intensity" in hf["peaks"])
                )
        if not is_integrated:
            return None

        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            hist_items = self.list_history_datasets(limit=200)
            pred_cands: list[tuple[int, dict[str, Any]]] = []
            for ds in hist_items:
                if ds.get("state") != "ok":
                    continue
                dname = str(ds.get("name") or "").strip().lower()
                ext = str(ds.get("extension") or "").strip().lower()
                if "predict" in dname and (
                    dname.endswith((".h5", ".hdf5")) or ext in ("h5", "hdf5")
                ):
                    hid_v = int(ds.get("hid") or 0)
                    # Prefer predictor datasets created at or just before target_hid, then newer datasets
                    rank = (
                        (target_hid - hid_v)
                        if (target_hid is not None and hid_v <= target_hid)
                        else (100000 + abs((target_hid or 0) - hid_v))
                    )
                    pred_cands.append((rank if target_hid is not None else -hid_v, ds))
            pred_cands.sort(key=lambda kv: kv[0])
            for _, cand_ds in pred_cands[:4]:
                cand_id = str(cand_ds.get("id") or "").strip()
                if not cand_id:
                    continue
                self.resolve_dataset_to_local_h5(cand_id)
                found = _discover_predictor_h5(peaks_h5_path)
                if found and os.path.isfile(found):
                    return found
        return None

    def connect(self) -> Any:
        if self._conn_helper is not None:
            return self._conn_helper
        if not self.is_connected:
            raise RuntimeError(
                "GALAXY_URL and GALAXY_API_KEY environment variables are required "
                "for live NDIP Galaxy execution."
            )
        from nova.galaxy import Connection

        conn = Connection(self.galaxy_url, self.galaxy_api_key)
        # Preserve explicit port (e.g. :9090 for local Planemo/Galaxy) stripped by nova.galaxy.Connection.__init__
        conn.galaxy_url = self.galaxy_url
        self._conn_helper = conn.connect()
        return self._conn_helper

    def get_datastore(self, history_name: str = "IMAGINE-X SubHKL") -> Any:
        helper = self.connect()
        store = helper.get_data_store(history_name, create=True)
        store_hid = str(getattr(store, "history_id", "") or "").strip()
        if store_hid and history_name == "IMAGINE-X SubHKL":
            if not getattr(self, "_launch_history_id", None) and self.history_id:
                self._launch_history_id = self.history_id
            self._subhkl_datastore_history_id = store_hid
            self.history_id = store_hid
            self._cached_history_datasets = None
            self._cached_history_datasets_ts = 0.0
        return store

    def _has_tool_on_galaxy(self, tool_id: str) -> bool:
        try:
            doc = self._api_request("GET", f"/api/tools/{tool_id}")
            return isinstance(doc, dict) and doc.get("id") == tool_id
        except _NOVA_RECOVERABLE_ERRORS:
            return False

    def _get_tool_version(self, tool_id: str) -> str:
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            doc = self._api_request("GET", f"/api/tools/{tool_id}")
            if isinstance(doc, dict):
                return str(doc.get("version") or "")
        return ""

    def _resolve_careless_merge_mode(
        self,
        params: dict[str, Any] | None = None,
        *,
        work_dir: str = "",
        merged_path: str = "",
        solve_path: str = "",
        integrated_path: str = "",
    ) -> str:
        """Choose ``'mono'`` whenever TOF data is available in workflow artifacts, else ``'poly'``."""
        from subhkl.viz.rainbow_zones import _h5_has_tof_data

        pdict = params or {}
        if _h5_has_tof_data(
            integrated_path,
            os.path.join(work_dir, "integrated.h5") if work_dir else None,
            os.path.join(work_dir, "integrator.h5") if work_dir else None,
            pdict.get("peaks_file"),
            pdict.get("images_file"),
            merged_path,
            os.path.join(work_dir, "merged.h5") if work_dir else None,
            solve_path,
            getattr(self, "replay_h5", None),
        ):
            return "mono"
        if bool(pdict.get("has_tof_data")):
            return "mono"
        explicit = (
            str(pdict.get("merge_mode") or pdict.get("mode") or "").strip().lower()
        )
        if explicit in ("mono", "poly"):
            return explicit
        return "poly"

    @staticmethod
    def is_explicit_galaxy_exec_mode() -> bool:
        """Return True when SUBHKL_EXEC_MODE explicitly selects Galaxy runner orchestration."""
        raw_mode = str(os.environ.get("SUBHKL_EXEC_MODE") or "").strip().lower()
        return raw_mode in ("galaxy", "remote", "0", "false", "no")

    def is_galaxy_exec_mode(self) -> bool:
        """Return True when pipeline subtasks execute via Galaxy runners rather than local GPU."""
        if self.is_explicit_galaxy_exec_mode():
            return True
        return bool(self.is_connected and not self._is_local_staging_mode())

    CLEANUP_SHARED_TOOL_ID = "imaginex_subhkl_cleanup_shared"

    def require_shared_dir_for_galaxy_mode(self, *, writable: bool = False) -> str:
        """Validate and resolve mandatory SUBHKL_SHARED_DIR when executing in Galaxy mode.

        By default the UI container only requires read-only (``R_OK | X_OK``) access;
        worker writers and local-to-shared staging pass ``writable=True``.
        """
        from subhkl.io.shared_storage import get_shared_dir_local

        local_shared = get_shared_dir_local(
            require=True,
            create=writable,
            writable=writable,
        )
        assert local_shared is not None
        return local_shared

    def _purge_galaxy_dataset_id(self, ds_id: str | None) -> None:
        """Best-effort delete/purge of a transient shared-pointer HDA in Galaxy History."""
        clean_id = str(ds_id or "").strip()
        if not clean_id or not self.is_connected:
            return
        hist_map: dict[str, str] = getattr(self, "_dataset_history_by_id", {})
        hid = str(hist_map.get(clean_id) or self.history_id or "").strip()
        if not hid:
            return
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            self._api_request(
                "PUT",
                f"/api/histories/{hid}/contents/{clean_id}",
                payload={"deleted": True, "purged": True},
            )
        self._cached_history_datasets = None

    def _dispatch_galaxy_shared_cleanup(
        self,
        targets: list[str],
        *,
        if_ephemeral: bool = False,
        wait: bool = False,
        job_record: dict[str, Any] | None = None,
    ) -> str | None:
        """Dispatch ``imaginex_subhkl_cleanup_shared`` on a worker with ``rw`` access when UI mount is ``ro``."""
        clean_targets: list[str] = []
        for raw_t in targets:
            t_str = str(raw_t or "").strip()
            if (
                t_str
                and "subhkl_shared_" in os.path.basename(t_str)
                and t_str not in clean_targets
            ):
                clean_targets.append(t_str)
        if not clean_targets or not self.is_connected:
            return None
        if not self._has_tool_on_galaxy(self.CLEANUP_SHARED_TOOL_ID):
            return None

        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            store = self.get_datastore("IMAGINE-X SubHKL")
            if not self.history_id and getattr(store, "history_id", None):
                self.history_id = store.history_id
            gi = store.nova_connection.galaxy_instance
            tool_inputs: dict[str, Any] = {
                "networkPath": ",".join(clean_targets),
                "ifEphemeral": bool(if_ephemeral),
            }
            res = gi.tools.run_tool(
                history_id=store.history_id,
                tool_id=self.CLEANUP_SHARED_TOOL_ID,
                tool_inputs=tool_inputs,
            )
            c_jid = str((res.get("jobs") or [{}])[0].get("id") or "").strip()
            c_outs = [
                str(o.get("id"))
                for o in (res.get("outputs") or [])
                if isinstance(o, dict) and o.get("id")
            ]
            if isinstance(job_record, dict) and c_jid:
                names = ", ".join(os.path.basename(p) for p in clean_targets)
                job_record["stdout"] = job_record.get("stdout", "") + (
                    f"[galaxy -> pulsar] Dispatched worker cleanup job {c_jid} for {names}\n"
                )

            def _reap_cleanup_job() -> None:
                if c_jid:
                    for _ in range(15):
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            jinfo = self._api_request("GET", f"/api/jobs/{c_jid}")
                            if isinstance(jinfo, dict) and jinfo.get("state") in (
                                "ok",
                                "error",
                                "deleted",
                                "paused",
                            ):
                                break
                        time.sleep(0.4)
                for oid in c_outs:
                    self._purge_galaxy_dataset_id(oid)

            if wait:
                _reap_cleanup_job()
            elif c_outs:
                threading.Thread(
                    target=_reap_cleanup_job,
                    name=f"subhkl-reap-cleanup-{c_jid or 'job'}",
                    daemon=True,
                ).start()
            return c_jid or None
        return None

    def release_shared_scan_intermediates(
        self,
        *,
        keep_work_dir: str | None = None,
        keep_paths: set[str] | None = None,
        job_record: dict[str, Any] | None = None,
        wait_remote: bool = False,
    ) -> list[str]:
        """Delete superseded shared intermediates in SUBHKL_SHARED_DIR (locally or via worker tool) and purge pointer HDAs."""
        from subhkl.io.shared_storage import (
            cleanup_shared_artifact,
            is_safe_shared_intermediate,
            to_shared_network_path,
        )

        keep_wd_abs = os.path.abspath(keep_work_dir) if keep_work_dir else None
        keep_abs_set = {
            os.path.abspath(os.path.realpath(p)) for p in (keep_paths or set()) if p
        }
        with _TEMP_CLEANUP_LOCK:
            candidates = list(_REGISTERED_SHARED_ARTIFACTS.items())

        removed_paths: list[str] = []
        remote_pending: list[tuple[str, str]] = []
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        ptr_map: dict[str, str] = getattr(self, "_shared_pointer_ds_by_path", {})

        for abs_p, meta in candidates:
            if abs_p in keep_abs_set:
                continue
            item_wd = str(meta.get("work_dir") or "")
            if (
                keep_wd_abs
                and item_wd
                and os.path.abspath(item_wd) == keep_wd_abs
                and not keep_abs_set
            ):
                continue
            with _TEMP_CLEANUP_LOCK:
                _REGISTERED_SHARED_ARTIFACTS.pop(abs_p, None)
            deleted = cleanup_shared_artifact(abs_p)
            if deleted:
                removed_paths.append(abs_p)
                if isinstance(job_record, dict):
                    job_record["stdout"] = job_record.get("stdout", "") + (
                        f"[subhkl] Cleaned up superseded shared intermediate '{os.path.basename(abs_p)}'\n"
                    )
            else:
                net_t = str(meta.get("network_path") or "").strip()
                if not net_t and is_safe_shared_intermediate(abs_p):
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        net_t = to_shared_network_path(abs_p)
                if net_t and (
                    os.path.lexists(abs_p)
                    or not os.path.isabs(net_t)
                    or "subhkl_shared_" in net_t
                ):
                    remote_pending.append((abs_p, net_t))
            ds_id = str(meta.get("dataset_id") or ptr_map.pop(abs_p, "") or "")
            ptr_map.pop(abs_p, None)
            known_map.pop(abs_p, None)
            if ds_id:
                for k_alias, v_id in list(known_map.items()):
                    if v_id == ds_id:
                        known_map.pop(k_alias, None)
                self._purge_galaxy_dataset_id(ds_id)
        if remote_pending:
            c_jid = self._dispatch_galaxy_shared_cleanup(
                [net_t for _abs_p, net_t in remote_pending],
                if_ephemeral=False,
                wait=wait_remote,
                job_record=job_record,
            )
            if c_jid:
                for abs_p, _net_t in remote_pending:
                    removed_paths.append(abs_p)
        self._dataset_id_by_path = known_map
        self._shared_pointer_ds_by_path = ptr_map
        if removed_paths:
            self._cached_history_datasets = None
            self._cached_history_datasets_ts = 0.0
        return removed_paths

    def cleanup_ephemeral_predictor_intermediate(
        self,
        pred_ds_id: str | None,
        *,
        work_dir: str | None = None,
        job_record: dict[str, Any] | None = None,
    ) -> None:
        """Delete ephemeral peak_predictor.h5 from SUBHKL_SHARED_DIR and purge its pointer HDA after Integrate."""
        from subhkl.io.shared_storage import (
            cleanup_shared_artifact,
            is_safe_shared_intermediate,
            to_shared_network_path,
        )

        clean_ds = str(pred_ds_id or "").strip()
        with _TEMP_CLEANUP_LOCK:
            matched_entries = [
                (k, dict(meta))
                for k, meta in _REGISTERED_SHARED_ARTIFACTS.items()
                if bool(meta.get("delete_after_use"))
                or (clean_ds and str(meta.get("dataset_id") or "") == clean_ds)
                or k.endswith("_peak_predictor.h5")
            ]
            for mk, _meta in matched_entries:
                _REGISTERED_SHARED_ARTIFACTS.pop(mk, None)
        remote_ephem: list[str] = []
        for mk, meta in matched_entries:
            if cleanup_shared_artifact(mk):
                if isinstance(job_record, dict):
                    job_record["stdout"] = job_record.get("stdout", "") + (
                        f"[subhkl] Deleted ephemeral shared predictor '{os.path.basename(mk)}'\n"
                    )
            elif os.path.lexists(mk):
                net_t = str(meta.get("network_path") or "").strip()
                if not net_t and is_safe_shared_intermediate(mk):
                    with contextlib.suppress(OSError, ValueError, RuntimeError):
                        net_t = to_shared_network_path(mk)
                if net_t:
                    remote_ephem.append(net_t)
        if remote_ephem:
            self._dispatch_galaxy_shared_cleanup(
                remote_ephem,
                if_ephemeral=False,
                wait=False,
                job_record=job_record,
            )
        if work_dir and os.path.isdir(work_dir):
            pred_link = os.path.join(work_dir, "predictor.h5")
            if os.path.islink(pred_link) and not os.path.exists(pred_link):
                with contextlib.suppress(OSError):
                    os.unlink(pred_link)
        if clean_ds:
            self._purge_galaxy_dataset_id(clean_ds)

    def shutdown_cleanup(self) -> None:
        """Cancel any active stage job and release session shared intermediates (locally or via cleanup worker)."""
        with contextlib.suppress(Exception):
            self.cancel_active_job()
        with contextlib.suppress(Exception):
            self.release_shared_scan_intermediates(wait_remote=True)

    def build_exit_restore_plan(
        self,
        *,
        current_h5: str | None = None,
        images_h5: str | None = None,
        payload: dict[str, Any] | None = None,
        mtz_overlay_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Snapshot how to restore the active step if the user cancels the Exit screen ('I changed my mind')."""
        if not self.is_connected:
            return {"mode": "local_deferred"}

        pl = payload if isinstance(payload, dict) else {}
        art_kind = str(pl.get("artifact_kind") or "").strip().lower()
        frame_index = int(
            pl.get("frame_index") or getattr(self, "current_frame_index", 0) or 0
        )
        if art_kind in ("", "empty") and not current_h5 and not images_h5:
            return {"mode": "empty"}

        self._cached_history_datasets = None
        self._cached_history_datasets_ts = 0.0
        hist_ds = list(self.list_history_datasets(limit=200) or [])
        hist_ds.sort(key=lambda d: int(d.get("hid") or 0), reverse=True)

        def _is_ok_non_pointer(ds: dict[str, Any]) -> bool:
            if ds.get("deleted") is True:
                return False
            if ds.get("state") and ds.get("state") != "ok":
                return False
            nm = str(ds.get("name") or "").strip().lower()
            return bool(ds.get("id")) and not nm.endswith(".ref.json")

        ok_ds = [d for d in hist_ds if _is_ok_non_pointer(d)]
        by_id = {str(d["id"]): d for d in ok_ds}
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        sig_map: dict[str, tuple[int, int, str]] = getattr(
            self, "_dataset_sig_by_path", {}
        )
        mtz_ds_id = str((mtz_overlay_info or {}).get("id") or "").strip()
        if mtz_ds_id and mtz_ds_id not in by_id:
            mtz_ds_id = ""

        # 1. Derived stages (solve / integrate / merge): restore thin HDF5 from Galaxy History (hydrates reduce on the fly)
        is_derived = art_kind in (
            "solve",
            "finder",
            "indexer",
            "integrator",
            "merged_careless",
        )
        if not is_derived and current_h5:
            bn_low = os.path.basename(str(current_h5)).lower()
            is_derived = any(k in bn_low for k in ("solve", "integrat", "predict"))

        if is_derived:
            restore_id = ""
            if current_h5:
                for p_key in (
                    os.path.abspath(str(current_h5)),
                    os.path.realpath(str(current_h5)),
                ):
                    cand_id = str(known_map.get(p_key) or "").strip()
                    if not cand_id and p_key in sig_map:
                        cand_id = str(sig_map[p_key][2] or "").strip()
                    if cand_id and cand_id in by_id:
                        cand_nm = str(by_id[cand_id].get("name") or "").lower()
                        if not cand_nm.endswith((".mtz", ".csv", ".html", ".json")):
                            restore_id = cand_id
                            break
            if not restore_id:
                want_int = art_kind in ("integrator", "merged_careless") or (
                    current_h5
                    and "integrat" in os.path.basename(str(current_h5)).lower()
                )
                for d in ok_ds:
                    nm = os.path.basename(str(d.get("name") or "")).lower()
                    if nm.endswith(
                        (".mtz", ".csv", ".html", ".json", ".nxs", ".nxs.h5")
                    ):
                        continue
                    if want_int and "integrat" in nm:
                        restore_id = str(d["id"])
                        break
                    if not want_int and ("solve" in nm or "index" in nm):
                        restore_id = str(d["id"])
                        break
            if restore_id:
                return {
                    "mode": "galaxy_dataset",
                    "dataset_id": restore_id,
                    "mtz_dataset_id": mtz_ds_id,
                    "frame_index": frame_index,
                    "artifact_kind": art_kind,
                }

        # 2. Step 1 (Reduce / merged.h5): check persisted merged.h5, local/facility NeXus recipe, or raw NeXus dataset in History
        for d in ok_ds:
            nm = os.path.basename(str(d.get("name") or "")).lower()
            if (
                any(k in nm for k in ("merged", "scan", "image"))
                and "careless" not in nm
                and nm.endswith((".h5", ".hdf5"))
            ):
                return {
                    "mode": "galaxy_dataset",
                    "dataset_id": str(d["id"]),
                    "frame_index": frame_index,
                    "artifact_kind": "merged",
                }

        src_for_recipe = images_h5 or current_h5
        if src_for_recipe and os.path.isfile(str(src_for_recipe)):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                recipe = self._extract_reduce_recipe_from_derived_h5(
                    str(src_for_recipe)
                )
                if recipe and recipe.get("nexus_files"):
                    return {
                        "mode": "reduce_recipe",
                        "params": recipe,
                        "frame_index": frame_index,
                    }

        for d in ok_ds:
            nm = str(d.get("name") or "").lower()
            ext = str(d.get("extension") or "").lower()
            if nm.endswith((".nxs.h5", ".nxs")) or ext in ("nxs", "nxs.h5"):
                return {
                    "mode": "galaxy_dataset",
                    "dataset_id": str(d["id"]),
                    "frame_index": 0,
                    "artifact_kind": "raw_nexus",
                }

        for d in ok_ds:
            nm = os.path.basename(str(d.get("name") or "")).lower()
            if nm.endswith((".h5", ".hdf5")) and any(
                k in nm for k in ("integrat", "solve", "index")
            ):
                return {
                    "mode": "galaxy_dataset",
                    "dataset_id": str(d["id"]),
                    "frame_index": frame_index,
                    "artifact_kind": art_kind,
                }

        return {"mode": "empty"}

    def execute_app_exit(
        self,
        *,
        phase: str = "confirm",
        current_h5: str | None = None,
        images_h5: str | None = None,
        payload: dict[str, Any] | None = None,
        mtz_overlay_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle two-step app exit ('prepare' on modal open, 'confirm' on tab close/shutdown).

        In Galaxy/NDIP mode, 'prepare' snapshots a Galaxy-history restore plan and immediately
        sweeps shared + staging intermediates while keeping the server alive so the user can still
        click 'I changed my mind' (reconstructing reduced frames on the fly from Galaxy History).
        In Local mode (no Galaxy History), 'prepare' cancels any active job but defers deleting
        files until 'confirm' (or process termination).
        """
        t0 = time.monotonic()
        norm_phase = str(phase or "confirm").strip().lower()
        restore_plan: dict[str, Any] | None = None
        if norm_phase == "prepare":
            with contextlib.suppress(Exception):
                restore_plan = self.build_exit_restore_plan(
                    current_h5=current_h5,
                    images_h5=images_h5,
                    payload=payload,
                    mtz_overlay_info=mtz_overlay_info,
                )

        cancelled_active = False
        cleaned_shared: list[str] = []
        cleaned_staging: list[str] = []
        with contextlib.suppress(Exception):
            cancel_res = self.cancel_active_job()
            cancelled_active = bool(
                isinstance(cancel_res, dict)
                and (cancel_res.get("cancelled") or cancel_res.get("job"))
            )

        galaxy_conn = bool(self.is_connected)
        defer_files = norm_phase == "prepare" and not galaxy_conn
        if not defer_files:
            with contextlib.suppress(Exception):
                cleaned_shared = self.release_shared_scan_intermediates(
                    wait_remote=True
                )
            with contextlib.suppress(Exception):
                cleaned_staging = cleanup_registered_temp_artifacts()

        elapsed_ms = round((time.monotonic() - t0) * 1000)
        out: dict[str, Any] = {
            "ok": True,
            "status": "prepared" if norm_phase == "prepare" else "exited",
            "phase": norm_phase,
            "deferred_cleanup": defer_files,
            "cancelled_active_job": cancelled_active,
            "cleaned_shared_count": len(cleaned_shared),
            "cleaned_staging_count": len(cleaned_staging),
            "galaxy_connected": galaxy_conn,
            "elapsed_ms": elapsed_ms,
        }
        if restore_plan is not None:
            out["restore_plan"] = restore_plan
        return out

    def _is_local_staging_mode(self) -> bool:
        """Return True when heavy subhkl stages should run inside local NVMe staging.

        Enabled when:
          - ``SUBHKL_EXEC_MODE`` is ``local`` / ``staging`` / ``1`` / ``true``; or
          - ``GALAXY_URL`` / ``GALAXY_API_KEY`` are absent; or
          - running inside an NDIP interactive GPU tool container (``EP_PATH`` set)
            with direct ``/SNS`` or ``/HFIR`` filesystem mounts.
        Explicit ``SUBHKL_EXEC_MODE=galaxy`` / ``remote`` / ``0`` / ``false`` forces
        remote Galaxy job execution for all stages.
        """
        raw_mode = str(os.environ.get("SUBHKL_EXEC_MODE") or "").strip().lower()
        if raw_mode in ("local", "staging", "1", "true", "yes"):
            return True
        if raw_mode in ("galaxy", "remote", "0", "false", "no"):
            return False
        if not self.is_connected:
            return True
        return bool(os.environ.get("EP_PATH")) and self._has_local_sns_hfir_fs()

    @staticmethod
    def _has_local_careless_runner() -> bool:
        """Return True if ``careless`` CLI or ``podman`` container runtime is available locally."""
        import shutil as _shutil

        return bool(_shutil.which("careless") or _shutil.which("podman"))

    def _should_run_stage_locally(
        self,
        stage_id: str,
        params: dict[str, Any],
    ) -> tuple[bool, str]:
        """Decide whether ``stage_id`` executes in local staging vs. remote NDIP Galaxy."""
        if not self.is_connected:
            return True, "GALAXY_URL / GALAXY_API_KEY not set"
        if not self._is_local_staging_mode():
            return False, ""
        if stage_id == "merge":
            # Keep Careless out of the subhkl container: when careless/podman is absent
            # locally, dispatch Stage 4 through Galaxy using compact reflections.mtz.
            if not self._has_local_careless_runner():
                return False, ""
            return True, "local staging execution (compact deliverables synced to NDIP)"
        if stage_id == "reduce" and not self._has_local_sns_hfir_fs():
            local_nxs = self._resolve_raw_nexus_inputs(params, _get_staging_root())
            if not local_nxs or not any(os.path.isfile(p) for p in local_nxs):
                return False, ""
        return True, "local GPU staging (compact deliverables synced to NDIP)"

    def _upload_compact_ndip_artifact(
        self,
        store: Any,
        local_path: str,
        file_ext: str,
        *,
        job_record: dict[str, Any] | None = None,
        max_bytes: int = _MAX_NDIP_COMPACT_UPLOAD_BYTES,
    ) -> str | None:
        """Upload a compact deliverable (``<= max_bytes``) to NDIP Galaxy History once per file state."""
        if not local_path or not os.path.isfile(local_path):
            return None
        try:
            st = os.stat(local_path)
        except OSError:
            return None
        if st.st_size <= 0 or st.st_size > max_bytes:
            if isinstance(job_record, dict):
                job_record["stdout"] = job_record.get("stdout", "") + (
                    f"[ndip] Skipping Galaxy upload for '{os.path.basename(local_path)}' "
                    f"({st.st_size / (1024 * 1024):.1f} MB > {max_bytes / (1024 * 1024):.0f} MB compact threshold)\n"
                )
            return None
        abs_p = os.path.abspath(local_path)
        real_p = os.path.realpath(local_path)
        rec = job_record if isinstance(job_record, dict) else {"stdout": ""}
        with self._upload_lock:
            sig_map = getattr(self, "_dataset_sig_by_path", {})
            prev = sig_map.get(abs_p) or sig_map.get(real_p)
            if prev and prev[0] == st.st_size and prev[1] == st.st_mtime_ns and prev[2]:
                return prev[2]
            ds_id = self._ensure_galaxy_h5_dataset(
                store,
                local_path,
                rec,
                force_upload=True,
                file_ext=file_ext,
            )
            if ds_id:
                sig_entry = (st.st_size, st.st_mtime_ns, ds_id)
                sig_map[abs_p] = sig_entry
                sig_map[real_p] = sig_entry
                self._dataset_sig_by_path = sig_map
            return ds_id

    def submit_galaxy_stage(
        self,
        stage_id: str,
        params: dict[str, Any] | None = None,
        on_complete_h5: Any = None,
    ) -> dict[str, Any]:
        """Submit a coarse NDIP workflow stage (local GPU staging or remote Galaxy/Pulsar)."""
        if stage_id not in self.STAGE_TOOLS:
            raise ValueError(f"Unknown pipeline stage: {stage_id}")
        params = dict(params or {})
        run_locally, local_reason = self._should_run_stage_locally(stage_id, params)
        if run_locally:
            return self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=local_reason,
            )

        if not self._is_local_staging_mode():
            self.require_shared_dir_for_galaxy_mode()

        init_job_id = f"nova-{int(time.time())}"
        job_record: dict[str, Any] = {
            "job_id": init_job_id,
            "client_job_id": init_job_id,
            "tool_id": self.STAGE_TOOLS[stage_id][0],
            "stage_id": stage_id,
            "mode": "galaxy",
            "state": "queued",
            "command_line": f"nova-galaxy -> {self.galaxy_url}",
            "stdout": (
                f"[nova -> galaxy] Connecting to {self.galaxy_url} (datastore: 'IMAGINE-X SubHKL')...\n"
            ),
            "stderr": "",
            "ladder_rung_idx": 0 if stage_id == "solve" else 4,
            "outputs": [],
            "updated_payload_ready": False,
            "started_at": time.time(),
            "hydrate_only": bool(params.get("hydrate_only")),
        }
        if params.get("hydrate_only"):
            job_record["work_dir"] = str(params.get("work_dir") or "")
            job_record["n_runs"] = len(params.get("nexus_files") or ())
        if stage_id == "solve":
            wl_r = params.get("wavelength_range")
            raw_bin = params.get("binning")
            solve_binning = (
                int(raw_bin)
                if raw_bin not in (None, "", "auto", "default") and int(raw_bin) >= 1
                else None
            )
            job_record["solve_params"] = {
                "instrument": str(params.get("instrument") or "CG4D"),
                "space_group": str(params.get("space_group") or "P 1"),
                "cell": params.get("cell"),
                "wavelength_range": (
                    (float(wl_r[0]), float(wl_r[1]))
                    if isinstance(wl_r, (list, tuple)) and len(wl_r) >= 2
                    else None
                ),
                "d_min": (
                    float(params["d_min"]) if params.get("d_min") is not None else None
                ),
                "binning": solve_binning,
            }
        self.active_job = job_record
        self._sync_job_timing(job_record)

        threading.Thread(
            target=self._dispatch_nova_galaxy_thread,
            args=(stage_id, params, job_record, on_complete_h5),
            daemon=True,
        ).start()

        return {
            "stage_id": stage_id,
            "mode": "ndip_galaxy_rest",
            "galaxy_url": self.galaxy_url,
            "history_id": self.history_id,
            "job": job_record,
            "status": "submitted",
        }

    @staticmethod
    def _terminate_local_container_or_proc(
        container_name: str | None,
        proc: subprocess.Popen[bytes] | None,
    ) -> bool:
        """Immediately kill and force-remove a local Podman container and/or subprocess."""
        import signal

        killed_container = False
        podman_bin = shutil.which("podman")
        if container_name and podman_bin:
            for subcmd, t_out in (
                ([podman_bin, "kill", str(container_name)], 5.0),
                ([podman_bin, "rm", "-f", "-i", "-t", "0", str(container_name)], 10.0),
            ):
                with contextlib.suppress(
                    OSError, subprocess.SubprocessError, ValueError, TypeError
                ):
                    res = subprocess.run(
                        subcmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=t_out,
                        check=False,
                    )
                    if res.returncode == 0:
                        killed_container = True

        if proc is not None and proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except OSError:
                with contextlib.suppress(OSError):
                    proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=0.3)
            if proc.poll() is None:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except OSError:
                    with contextlib.suppress(OSError):
                        proc.kill()

        return killed_container

    def cancel_active_job(self, stage_id: str | None = None) -> dict[str, Any]:
        """Cancel the currently running NDIP Galaxy or local container job."""
        aj = self.active_job
        if not isinstance(aj, dict):
            return {"ok": True, "status": "idle", "message": "No active job to cancel"}

        aj["cancel_requested"] = True
        prev_state = str(aj.get("state") or "")
        aj["state"] = "deleted"
        self._sync_job_timing(aj)
        aj["stdout"] = (
            aj.get("stdout", "")
            + "\n[subhkl] Stop requested by user — cancelling job...\n"
        )[-24000:]

        cancelled_galaxy_ids: list[str] = []
        gx_ids: list[str] = list(aj.get("galaxy_job_ids") or [])
        cur_jid = str(aj.get("job_id") or "")
        if (
            cur_jid
            and not cur_jid.startswith(("nova-", "gpu-"))
            and cur_jid not in gx_ids
        ):
            gx_ids.append(cur_jid)

        for jid in gx_ids:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                self._api_request("DELETE", f"/api/jobs/{jid}")
                cancelled_galaxy_ids.append(jid)

        procs_map: dict[str, subprocess.Popen[bytes]] = getattr(
            self, "_active_procs", {}
        )
        proc = procs_map.get(cur_jid) or getattr(self, "_active_proc", None)
        cname = (
            str(
                aj.get("container_name")
                or getattr(self, "_active_container_name", None)
                or ""
            ).strip()
            or None
        )

        killed_container = self._terminate_local_container_or_proc(cname, proc)
        if cname:
            aj["stdout"] = (
                aj.get("stdout", "") + f"[container] Stopped container {cname}\n"
            )[-24000:]

        return {
            "ok": True,
            "status": "cancelled",
            "stage_id": aj.get("stage_id") or stage_id,
            "previous_state": prev_state,
            "cancelled_job_id": cur_jid,
            "cancelled_galaxy_jobs": cancelled_galaxy_ids,
            "cancelled_container": cname if killed_container else None,
            "job": aj,
        }

    def _stamp_merged_metadata(self, h5_path: str, job_record: dict[str, Any]) -> None:
        if not h5_path or not os.path.isfile(h5_path):
            return
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py
            import numpy as np

            params = job_record.get("params") or {}
            inst_val = str(
                job_record.get("instrument") or params.get("instrument") or ""
            ).strip()
            sg_val = str(params.get("space_group") or "").strip()
            raw_cell = params.get("cell")
            if not (isinstance(raw_cell, (list, tuple)) and len(raw_cell) == 6) and all(
                k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
            ):
                raw_cell = [
                    float(params["a"]),
                    float(params["b"]),
                    float(params["c"]),
                    float(params["alpha"]),
                    float(params["beta"]),
                    float(params["gamma"]),
                ]
            with h5py.File(h5_path, "r+") as hf:
                if inst_val:
                    hf.attrs["instrument"] = inst_val
                    if "instrument" not in hf:
                        hf["instrument"] = np.bytes_(inst_val)
                    elif (
                        isinstance(hf["instrument"], h5py.Group)
                        and "name" not in hf["instrument"]
                    ):
                        hf["instrument/name"] = np.bytes_(inst_val)
                if sg_val and "space_group" not in hf:
                    hf["space_group"] = np.bytes_(sg_val)
                raw_nxs_list = (
                    job_record.get("raw_nxs") or params.get("nexus_files") or []
                )
                if isinstance(raw_nxs_list, str):
                    raw_nxs_list = [
                        s.strip() for s in raw_nxs_list.split(",") if s.strip()
                    ]
                clean_src_paths = [
                    str(p).strip() for p in raw_nxs_list if str(p).strip()
                ]
                if clean_src_paths:
                    if "source_files" in hf:
                        del hf["source_files"]
                    dt_src = h5py.string_dtype(encoding="utf-8")
                    hf.create_dataset(
                        "source_files", data=clean_src_paths, dtype=dt_src
                    )
                if (
                    clean_src_paths
                    and "files" in hf
                    and len(hf["files"]) == len(clean_src_paths)
                ):
                    orig_names = [os.path.basename(p) for p in clean_src_paths]
                    del hf["files"]
                    dt_str = h5py.string_dtype(encoding="utf-8")
                    hf.create_dataset("files", data=orig_names, dtype=dt_str)
                if (
                    isinstance(raw_cell, (list, tuple))
                    and len(raw_cell) == 6
                    and "sample/a" not in hf
                ):
                    grp = hf.require_group("sample")
                    for idx_c, k_c in enumerate(
                        ("a", "b", "c", "alpha", "beta", "gamma")
                    ):
                        if k_c not in grp:
                            grp[k_c] = float(raw_cell[idx_c])
                if sg_val and "sample/space_group" not in hf:
                    grp = hf.require_group("sample")
                    grp["space_group"] = np.bytes_(sg_val)

    def _stamp_derived_h5_provenance(
        self,
        h5_path: str,
        *,
        merged_path: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Copy raw NeXus provenance (`source_files`, `files`, `file_offsets`, cell, instrument) into `solve.h5` / `integrated.h5`."""
        if not h5_path or not os.path.isfile(h5_path):
            return
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py
            import numpy as np

            p_dict = params or {}
            src_files_vals: list[str] = []
            files_vals: list[str] = []
            file_offsets_arr: Any = None
            merged_inst = ""
            merged_wl: Any = None
            if merged_path and os.path.isfile(merged_path):
                with h5py.File(merged_path, "r") as mf:
                    if "source_files" in mf:
                        src_files_vals = [
                            x.decode("utf-8", "ignore")
                            if isinstance(x, (bytes, np.bytes_))
                            else str(x)
                            for x in list(mf["source_files"][()])
                        ]
                    if "files" in mf:
                        files_vals = [
                            x.decode("utf-8", "ignore")
                            if isinstance(x, (bytes, np.bytes_))
                            else str(x)
                            for x in list(mf["files"][()])
                        ]
                    if "file_offsets" in mf:
                        file_offsets_arr = np.asarray(mf["file_offsets"][()])
                    raw_mi = mf.attrs.get("instrument")
                    if raw_mi is not None:
                        merged_inst = (
                            raw_mi.decode("utf-8", "ignore")
                            if isinstance(raw_mi, (bytes, np.bytes_))
                            else str(raw_mi)
                        ).strip()
                    if "instrument/wavelength" in mf:
                        merged_wl = np.asarray(
                            mf["instrument/wavelength"][()], dtype=np.float64
                        )

            with h5py.File(h5_path, "r+") as hf:
                dt_str = h5py.string_dtype(encoding="utf-8")
                if src_files_vals and "source_files" not in hf:
                    hf.create_dataset("source_files", data=src_files_vals, dtype=dt_str)
                if files_vals and "files" not in hf:
                    hf.create_dataset("files", data=files_vals, dtype=dt_str)
                if file_offsets_arr is not None and "file_offsets" not in hf:
                    hf.create_dataset("file_offsets", data=file_offsets_arr)
                inst_s = str(p_dict.get("instrument") or merged_inst or "").strip()
                if inst_s and "instrument" not in hf.attrs:
                    hf.attrs["instrument"] = inst_s
                if merged_wl is not None and "instrument/wavelength" not in hf:
                    if "instrument" in hf and not isinstance(
                        hf["instrument"], h5py.Group
                    ):
                        del hf["instrument"]
                    igrp = hf.require_group("instrument")
                    igrp.create_dataset("wavelength", data=merged_wl)
                sg_s = str(p_dict.get("space_group") or "").strip()
                if sg_s and "sample/space_group" not in hf and "space_group" not in hf:
                    sgrp = hf.require_group("sample")
                    sgrp["space_group"] = np.bytes_(sg_s)

    def _extract_reduce_recipe_from_derived_h5(
        self,
        h5_path: str,
        *,
        ds_info: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract raw NeXus paths + reduction metadata needed to regenerate ``merged.h5`` on the fly."""
        import glob as _glob
        import re as _re

        if not h5_path or not os.path.isfile(h5_path):
            return None

        inst_val = ""
        sg_val = ""
        wl_vals: list[float] = []
        raw_cell: list[float] | None = None
        src_candidates: list[str] = []
        file_names: list[str] = []
        src_attr_str = ""

        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            import h5py
            import numpy as np

            with h5py.File(h5_path, "r") as hf:
                if "images" in hf:
                    return None
                if "source_files" in hf:
                    src_candidates = [
                        (
                            x.decode("utf-8", "ignore")
                            if isinstance(x, (bytes, np.bytes_))
                            else str(x)
                        ).strip()
                        for x in list(hf["source_files"][()])
                    ]
                    src_candidates = [x for x in src_candidates if x]
                if "files" in hf:
                    file_names = [
                        (
                            x.decode("utf-8", "ignore")
                            if isinstance(x, (bytes, np.bytes_))
                            else str(x)
                        ).strip()
                        for x in list(hf["files"][()])
                    ]
                    file_names = [x for x in file_names if x]
                raw_sa = hf.attrs.get("source")
                if raw_sa is not None:
                    src_attr_str = (
                        raw_sa.decode("utf-8", "ignore")
                        if isinstance(raw_sa, (bytes, np.bytes_))
                        else str(raw_sa)
                    ).strip()
                if "instrument" in hf.attrs:
                    iv = hf.attrs["instrument"]
                    inst_val = (
                        iv.decode("utf-8", "ignore")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                elif "instrument/name" in hf:
                    iv = hf["instrument/name"][()]
                    if hasattr(iv, "flat") and iv.size > 0:
                        iv = iv.flat[0]
                    inst_val = (
                        iv.decode("utf-8", "ignore")
                        if isinstance(iv, (bytes, np.bytes_))
                        else str(iv)
                    ).strip()
                if "instrument/wavelength" in hf:
                    w_arr = np.asarray(
                        hf["instrument/wavelength"][()], dtype=float
                    ).ravel()
                    if w_arr.size >= 2 and np.all(np.isfinite(w_arr[:2])):
                        wl_vals = [float(w_arr[0]), float(w_arr[1])]
                if all(
                    f"sample/{k}" in hf
                    for k in ("a", "b", "c", "alpha", "beta", "gamma")
                ):
                    raw_cell = [
                        float(np.asarray(hf[f"sample/{k}"][()]).flat[0])
                        for k in ("a", "b", "c", "alpha", "beta", "gamma")
                    ]
                elif "sample/B" in hf:
                    from subhkl.viz.rainbow_zones import cell_params_from_B

                    B = np.asarray(hf["sample/B"][()], dtype=np.float64)
                    if B.shape == (3, 3):
                        raw_cell = list(cell_params_from_B(B))
                for sk in ("sample/space_group", "space_group"):
                    if sk in hf:
                        sv = hf[sk][()]
                        if hasattr(sv, "flat") and sv.size > 0:
                            sv = sv.flat[0]
                        sg_val = (
                            sv.decode("utf-8", "ignore")
                            if isinstance(sv, (bytes, np.bytes_))
                            else str(sv)
                        ).strip()
                        if sg_val:
                            break

        def _facility_roots_for_inst(inst_name: str) -> list[str]:
            iu = inst_name.strip().upper()
            facs = (
                ["SNS", "HFIR"]
                if iu in ("MANDI", "TOPAZ", "CORELLI", "SNAP")
                else ["HFIR", "SNS"]
            )
            roots: list[str] = []
            for fc in facs:
                r = self._facility_mount_root(fc).rstrip("/")
                if r and os.path.isdir(r) and r not in roots:
                    roots.append(r)
            return roots

        def _locate_basename_on_fs(bname: str, inst_hint: str) -> str | None:
            bn_only = os.path.basename(bname)
            m_ib = _re.match(
                r"^([A-Za-z0-9]+)_([0-9]+)(?:\.reduce)?\.(?:nxs\.h5|nxs|h5|hdf5)$",
                bn_only,
                flags=_re.IGNORECASE,
            )
            inst_eff = (m_ib.group(1) if m_ib else inst_hint).strip().upper()
            if not inst_eff:
                return None
            probe_names: list[str] = []
            if m_ib:
                run_tok = m_ib.group(2)
                for ext_s in (".nxs.h5", ".nxs"):
                    cand_n = f"{inst_eff}_{run_tok}{ext_s}"
                    if cand_n not in probe_names:
                        probe_names.append(cand_n)
            if bn_only not in probe_names and bn_only.lower().endswith(
                (".nxs.h5", ".nxs")
            ):
                probe_names.append(bn_only)
            if not probe_names:
                return None
            for root_d in _facility_roots_for_inst(inst_eff):
                inst_d = os.path.join(root_d, inst_eff)
                if not os.path.isdir(inst_d):
                    continue
                for p_name in probe_names:
                    matches = sorted(
                        _glob.glob(os.path.join(inst_d, "IPTS-*", "nexus", p_name))
                    )
                    for m_p in matches:
                        if os.path.isfile(m_p) and os.access(m_p, os.R_OK):
                            return m_p
            return None

        resolved_nxs: list[str] = []
        allow_remote_facility_paths = self.is_galaxy_exec_mode()
        # 1. Explicit full paths or basenames (including transient *.reduce.h5 paths) from `source_files` or `files`
        for cand_list in (src_candidates, file_names):
            if resolved_nxs:
                break
            tmp_list: list[str] = []
            for p_raw in cand_list:
                remapped = self._remap_to_facility_root(p_raw)
                if (
                    remapped
                    and os.path.isfile(remapped)
                    and os.access(remapped, os.R_OK)
                    and not remapped.lower().endswith(".reduce.h5")
                ):
                    tmp_list.append(os.path.abspath(remapped))
                else:
                    loc_hit = _locate_basename_on_fs(p_raw, inst_val)
                    if loc_hit:
                        tmp_list.append(os.path.abspath(loc_hit))
                    elif (
                        allow_remote_facility_paths
                        and _re.match(
                            r"^/(SNS|HFIR)/.+", str(p_raw or ""), _re.IGNORECASE
                        )
                        and str(p_raw).lower().endswith((".nxs.h5", ".nxs"))
                        and not str(p_raw).lower().endswith(".reduce.h5")
                    ):
                        tmp_list.append(str(p_raw))
            if tmp_list and len(tmp_list) == len(cand_list):
                resolved_nxs = tmp_list

        # 2. Fallback for pre-existing NDIP datasets: infer <INST>_<RUN1>[_<INST>_<RUN2>] from attrs['source'] or dataset name
        if not resolved_nxs:
            hint_texts = [
                src_attr_str,
                str((ds_info or {}).get("name") or ""),
                os.path.basename(h5_path),
            ]
            for htxt in hint_texts:
                if not htxt:
                    continue
                bn_h = os.path.basename(htxt)
                m_rng = _re.search(
                    r"([A-Za-z0-9]+)_([0-9]{3,})(?:_(?:\1_)?([0-9]{3,}))?",
                    bn_h,
                )
                if not m_rng:
                    continue
                inst_cand = m_rng.group(1).upper()
                if inst_cand.lower() in ("subhkl", "gx", "work", "dataset"):
                    continue
                r1 = int(m_rng.group(2))
                r2 = int(m_rng.group(3)) if m_rng.group(3) else r1
                if r2 < r1 or (r2 - r1) > 500:
                    continue
                first_hit = _locate_basename_on_fs(
                    f"{inst_cand}_{r1}.nxs.h5", inst_cand
                ) or _locate_basename_on_fs(f"{inst_cand}_{r1}.nxs", inst_cand)
                if not first_hit:
                    continue
                if not inst_val:
                    inst_val = inst_cand
                if r1 == r2:
                    resolved_nxs = [os.path.abspath(first_hit)]
                    break
                nxs_dir = os.path.dirname(first_hit)
                range_hits: list[str] = []
                for r_num in range(r1, r2 + 1):
                    for ext_s in (".nxs.h5", ".nxs"):
                        cp = os.path.join(nxs_dir, f"{inst_cand}_{r_num}{ext_s}")
                        if os.path.isfile(cp) and os.access(cp, os.R_OK):
                            range_hits.append(os.path.abspath(cp))
                            break
                if range_hits:
                    resolved_nxs = range_hits
                    break

        if not resolved_nxs:
            return None

        inst_final = (inst_val or "CG4D").strip().upper()
        if not wl_vals or len(wl_vals) < 2:
            wl_vals = (
                [2.0, 4.0]
                if inst_final == "MANDI"
                else ([0.4, 3.5] if inst_final == "TOPAZ" else [2.8, 4.5])
            )
        if not raw_cell or len(raw_cell) < 6:
            raw_cell = [68.7, 69.8, 156.9, 90.0, 90.0, 90.0]
        return {
            "nexus_files": resolved_nxs,
            "instrument": inst_final,
            "wavelength_range": [float(wl_vals[0]), float(wl_vals[1])],
            "wavelength_min": float(wl_vals[0]),
            "wavelength_max": float(wl_vals[1]),
            "cell": [float(x) for x in raw_cell[:6]],
            "a": float(raw_cell[0]),
            "b": float(raw_cell[1]),
            "c": float(raw_cell[2]),
            "alpha": float(raw_cell[3]),
            "beta": float(raw_cell[4]),
            "gamma": float(raw_cell[5]),
            "space_group": sg_val or "P 1",
        }

    def _ensure_reflections_mtz_from_integrated(
        self,
        work_dir: str,
        *,
        peaks_file: str | None = None,
        space_group: str = "",
        job_record: dict[str, Any] | None = None,
    ) -> str | None:
        """Ensure ``work_dir/reflections.mtz`` exists, exporting directly from ``integrated.h5`` if needed."""
        if not work_dir or not os.path.isdir(work_dir):
            return None
        mtz_dest = os.path.join(work_dir, "reflections.mtz")
        if os.path.isfile(mtz_dest) and os.path.getsize(mtz_dest) > 0:
            return mtz_dest

        cand_int_paths: list[str] = [os.path.join(work_dir, "integrated.h5")]
        if peaks_file and os.path.isfile(str(peaks_file)):
            cand_int_paths.append(str(peaks_file))
            alt_mtz = os.path.join(
                os.path.dirname(os.path.abspath(str(peaks_file))), "reflections.mtz"
            )
            if os.path.isfile(alt_mtz) and os.path.getsize(alt_mtz) > 0:
                self._link_or_copy_file(alt_mtz, mtz_dest)
                _chmod_world_rw(mtz_dest, is_dir=False)
                return mtz_dest

        src_int = ""
        for cp in cand_int_paths:
            if not os.path.isfile(cp):
                continue
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py

                with h5py.File(cp, "r") as hf:
                    if "peaks/intensity" in hf or "peaks/hkl" in hf:
                        src_int = cp
                        break
        if not src_int:
            return None

        pred_cand = os.path.join(work_dir, "predictor.h5")
        solve_cand = os.path.join(work_dir, "solve.h5")
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            from subhkl.io.export import MTZExporter

            exp = MTZExporter(
                src_int,
                space_group=(space_group or None),
                predictions_file=(pred_cand if os.path.isfile(pred_cand) else None),
                corrections_file=(solve_cand if os.path.isfile(solve_cand) else None),
            )
            exp.write_mtz(mtz_dest)
            _chmod_world_rw(mtz_dest, is_dir=False)
            if isinstance(job_record, dict):
                job_record["stdout"] = job_record.get("stdout", "") + (
                    f"[subhkl] Exported reflections.mtz from {os.path.basename(src_int)} -> {mtz_dest}\n"
                )
            return mtz_dest if os.path.isfile(mtz_dest) else None
        return None

    def _maybe_trigger_hydrate_reduce(
        self,
        vm: Any,
        vm_lock: Any,
        derived_h5_path: str,
        *,
        ds_info: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Regenerate missing ``merged.h5`` detector image frames in the background when loading derived HDF5s."""
        if not derived_h5_path or not os.path.isfile(derived_h5_path):
            return None
        recipe = self._extract_reduce_recipe_from_derived_h5(
            derived_h5_path, ds_info=ds_info
        )
        if not recipe or not recipe.get("nexus_files"):
            return None

        work_d, _ = self._resolve_work_dir_and_merged(derived_h5_path)
        aj = self.active_job
        if isinstance(aj, dict) and aj.get("state") in ("running", "queued", "new"):
            if aj.get("hydrate_only") and os.path.abspath(
                str(aj.get("work_dir") or "")
            ) == os.path.abspath(work_d):
                return {
                    "job_id": aj.get("job_id"),
                    "client_job_id": aj.get("client_job_id") or aj.get("job_id"),
                    "mode": aj.get("mode"),
                    "state": aj.get("state"),
                    "n_runs": int(aj.get("n_runs") or len(recipe["nexus_files"])),
                    "hydrate_only": True,
                }
            return None

        target_replay_abs = os.path.abspath(derived_h5_path)

        def _on_hydrated(new_merged_h5: str) -> None:
            if not new_merged_h5 or not os.path.isfile(new_merged_h5):
                return
            with vm_lock, contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                if hasattr(vm, "data") and hasattr(vm, "refresh_payload"):
                    cur_rep = os.path.abspath(
                        str(getattr(vm.data, "peaks_file", "") or "")
                    )
                    cur_wd = (
                        self._resolve_work_dir_and_merged(cur_rep)[0] if cur_rep else ""
                    )
                    if cur_rep == target_replay_abs or (
                        cur_wd and os.path.abspath(cur_wd) == os.path.abspath(work_d)
                    ):
                        vm.data.images_file = new_merged_h5
                        cur_f = int(
                            getattr(
                                self,
                                "current_frame_index",
                                getattr(vm, "frame_index", 0),
                            )
                            or 0
                        )
                        vm.refresh_payload(frame_index=cur_f, preserve_params=True)
                else:
                    cur_rep = os.path.abspath(str(getattr(vm, "replay_h5", "") or ""))
                    cur_wd = (
                        self._resolve_work_dir_and_merged(cur_rep)[0] if cur_rep else ""
                    )
                    if cur_rep == target_replay_abs or (
                        cur_wd and os.path.abspath(cur_wd) == os.path.abspath(work_d)
                    ):
                        vm.payload_kwargs["images_h5"] = new_merged_h5
                        cur_f = int(getattr(self, "current_frame_index", 0) or 0)
                        vm.reload(images_h5=new_merged_h5, frame_index=cur_f)

        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            hydrate_params = dict(recipe)
            hydrate_params["hydrate_only"] = True
            hydrate_params["work_dir"] = work_d
            if self.is_galaxy_exec_mode():
                sub_res = self.submit_galaxy_stage(
                    "reduce",
                    hydrate_params,
                    on_complete_h5=_on_hydrated,
                )
                job_rec = (
                    sub_res.get("job")
                    if isinstance(sub_res.get("job"), dict)
                    else sub_res
                )
            else:
                job_rec = self._submit_local_subhkl_stage(
                    "reduce",
                    hydrate_params,
                    on_complete_h5=_on_hydrated,
                    fallback_reason=(
                        f"hydrating detector image frames on the fly from {len(recipe['nexus_files'])} raw NeXus run(s)"
                    ),
                    preserve_downstream=True,
                )
            return {
                "job_id": job_rec.get("job_id"),
                "client_job_id": job_rec.get("client_job_id") or job_rec.get("job_id"),
                "mode": job_rec.get("mode"),
                "state": job_rec.get("state", "running"),
                "n_runs": len(recipe["nexus_files"]),
                "hydrate_only": True,
            }
        return None

    def _normalize_galaxy_job_outputs(
        self,
        raw_outputs: list[Any],
        stage_id: str = "",
    ) -> list[dict[str, Any]]:
        """Normalize Galaxy ``run_tool`` output descriptors without mislabeling non-HDF5 datasets as ``h5``."""
        from subhkl.viz.rainbow_zones import infer_dataset_extension

        norm: list[dict[str, Any]] = []
        st_low = str(stage_id or "").strip().lower()
        for d in raw_outputs or []:
            if not isinstance(d, dict):
                continue
            out_name = str(d.get("output_name") or "").strip()
            label = str(d.get("name") or "").strip()
            primary_name = out_name or label
            raw_ext = d.get("file_ext") or d.get("extension")
            ext = infer_dataset_extension(label or out_name, raw_ext)
            if ext == "data" and out_name and out_name != label:
                ext = infer_dataset_extension(out_name, raw_ext)
            if ext == "data" and not raw_ext and st_low != "merge":
                ext = "h5"
            norm.append(
                {
                    "id": d.get("id"),
                    "name": primary_name,
                    "output_name": out_name or primary_name,
                    "label": label or primary_name,
                    "extension": ext,
                }
            )
        return norm

    def _reconcile_galaxy_job_outputs(
        self,
        current_outputs: list[Any],
        jinfo_outputs: Any,
        stage_id: str = "",
    ) -> list[dict[str, Any]]:
        """Merge ``/api/jobs/{id}`` output-name mappings into ``job_record['outputs']``."""
        from subhkl.viz.rainbow_zones import infer_dataset_extension

        st_low = str(stage_id or "").strip().lower()
        merged: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for item in current_outputs or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            oid = str(entry.get("id") or "").strip()
            nm = str(
                entry.get("label")
                or entry.get("name")
                or entry.get("output_name")
                or ""
            ).strip()
            raw_ext = entry.get("extension")
            if (
                st_low == "merge"
                and raw_ext == "h5"
                and not nm.lower().endswith((".h5", ".hdf5"))
            ):
                raw_ext = None
            inferred = infer_dataset_extension(nm, raw_ext)
            if inferred == "data" and entry.get("output_name"):
                inferred = infer_dataset_extension(
                    str(entry.get("output_name")), raw_ext
                )
            if inferred != "data" or not raw_ext:
                entry["extension"] = (
                    "h5"
                    if (inferred == "data" and not raw_ext and st_low != "merge")
                    else inferred
                )
            merged.append(entry)
            if oid:
                by_id[oid] = entry

        if isinstance(jinfo_outputs, dict):
            for out_key, out_val in jinfo_outputs.items():
                if not isinstance(out_val, dict):
                    continue
                oid = str(out_val.get("id") or "").strip()
                key_s = str(out_key or "").strip()
                if not oid or not key_s:
                    continue
                key_ext = infer_dataset_extension(key_s, None)
                if oid in by_id:
                    existing = by_id[oid]
                    if not existing.get("output_name"):
                        existing["output_name"] = key_s
                    if not existing.get("name"):
                        existing["name"] = key_s
                    cur_ext = str(existing.get("extension") or "").lower()
                    if key_ext != "data" and (
                        cur_ext in ("", "data", "auto", "binary")
                        or (st_low == "merge" and cur_ext == "h5")
                    ):
                        existing["extension"] = key_ext
                else:
                    ext_new = (
                        "h5" if (key_ext == "data" and st_low != "merge") else key_ext
                    )
                    new_entry = {
                        "id": oid,
                        "name": key_s,
                        "output_name": key_s,
                        "label": key_s,
                        "extension": ext_new,
                    }
                    merged.append(new_entry)
                    by_id[oid] = new_entry
        return merged

    @staticmethod
    def _match_careless_merge_output_name(out_d: dict[str, Any]) -> str | None:
        """Map a Galaxy ``imaginex_subhkl_merge`` output descriptor to its canonical local filename."""
        candidates = [
            str(out_d.get("output_name") or "").strip().lower(),
            str(out_d.get("label") or "").strip().lower(),
            str(out_d.get("name") or "").strip().lower(),
        ]
        exact_map = {
            "merged_mtz": "merged.mtz",
            "merged.mtz": "merged.mtz",
            "predictions_mtz": "predictions.mtz",
            "predictions.mtz": "predictions.mtz",
            "xval_mtz": "xval.mtz",
            "xval.mtz": "xval.mtz",
            "metrics_json": "careless_metrics.json",
            "careless_metrics_json": "careless_metrics.json",
            "careless_metrics.json": "careless_metrics.json",
            "merge_table_csv": "merge_table.csv",
            "merge_table.csv": "merge_table.csv",
            "merge_table_html": "merge_table.html",
            "merge_table.html": "merge_table.html",
        }
        for tok in candidates:
            if tok in exact_map:
                return exact_map[tok]
        for tok in candidates:
            if not tok:
                continue
            if "careless_metrics" in tok or tok.endswith("metrics_json"):
                return "careless_metrics.json"
            for canon in (
                "merged.mtz",
                "predictions.mtz",
                "xval.mtz",
                "merge_table.csv",
                "merge_table.html",
            ):
                if canon in tok or canon.replace(".", "_") in tok:
                    return canon
        return None

    def _prepare_writable_merge_h5_target(
        self,
        work_dir: str,
        job_record: dict[str, Any],
    ) -> str | None:
        """Ensure ``work_dir/integrated.h5`` is a local writable HDF5 file before Careless stamping."""
        import shutil as _shutil

        int_target = os.path.join(work_dir, "integrated.h5")
        if os.path.islink(int_target) and not os.path.isfile(int_target):
            with contextlib.suppress(OSError):
                os.unlink(int_target)

        params = job_record.get("params") or {}
        if not os.path.isfile(int_target):
            for cand in (
                os.path.join(work_dir, "integrator.h5"),
                str(params.get("peaks_file") or ""),
                os.path.join(work_dir, "solve.h5"),
            ):
                if (
                    cand
                    and os.path.isfile(cand)
                    and not cand.lower().endswith(
                        (".mtz", ".json", ".csv", ".html", ".txt", ".log")
                    )
                ):
                    with contextlib.suppress(OSError):
                        _shutil.copy2(os.path.realpath(cand), int_target)
                        _chmod_world_rw(int_target, is_dir=False)
                        break

        if os.path.isfile(int_target) and (
            os.path.islink(int_target) or not os.access(int_target, os.W_OK)
        ):
            src_real = os.path.realpath(int_target)
            tmp_int = os.path.join(work_dir, f".integrated_merge_{os.getpid()}.h5.tmp")
            try:
                _shutil.copy2(src_real, tmp_int)
                _chmod_world_rw(tmp_int, is_dir=False)
                os.replace(tmp_int, int_target)
            except OSError:
                with contextlib.suppress(OSError):
                    if os.path.exists(tmp_int):
                        os.unlink(tmp_int)

        if not os.path.isfile(int_target):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py
                import numpy as np

                inst_s = str(params.get("instrument") or "CG4D")
                sg_s = str(params.get("space_group") or "P 1")
                cell_raw = params.get("cell") or [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
                wl_raw = params.get("wavelength_range") or [1.0, 4.5]
                merged_mtz_p = os.path.join(work_dir, "merged.mtz")
                if os.path.isfile(merged_mtz_p):
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        import gemmi

                        mtz_c = gemmi.read_mtz_file(merged_mtz_p)
                        cell_raw = [
                            float(mtz_c.cell.a),
                            float(mtz_c.cell.b),
                            float(mtz_c.cell.c),
                            float(mtz_c.cell.alpha),
                            float(mtz_c.cell.beta),
                            float(mtz_c.cell.gamma),
                        ]
                        if mtz_c.spacegroup is not None:
                            sg_s = mtz_c.spacegroup.xhm()
                with h5py.File(int_target, "w") as hf:
                    hf["instrument"] = inst_s.encode("utf-8")
                    hf["wavelengths"] = np.asarray(wl_raw, dtype=np.float64)
                    sg_grp = hf.require_group("sample")
                    sg_grp["space_group"] = sg_s.encode("utf-8")
                    sg_grp["cell"] = np.asarray(cell_raw, dtype=np.float64)
                _chmod_world_rw(int_target, is_dir=False)

        return int_target if os.path.isfile(int_target) else None

    def _stamp_careless_merge_into_h5(
        self,
        integrated_h5_path: str,
        work_dir: str,
        job_record: dict[str, Any],
    ) -> None:
        """Persist Careless merge metrics and per-observation residuals/scales into ``integrated.h5``."""
        import gemmi
        import h5py
        import numpy as np

        metrics_path = os.path.join(work_dir, "careless_metrics.json")
        pred_mtz_path = os.path.join(work_dir, "predictions.mtz")
        merged_mtz_path = os.path.join(work_dir, "merged.mtz")
        xval_mtz_path = os.path.join(work_dir, "xval.mtz")
        metrics_raw = "{}"
        metrics_doc: dict[str, Any] = {}
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics_raw = fh.read()
                try:
                    metrics_doc = json.loads(metrics_raw)
                except ValueError:
                    metrics_doc = {}

        if os.path.isfile(merged_mtz_path):
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                mtz_m = gemmi.read_mtz_file(merged_mtz_path)
                mcols = {c.label: np.array(c, dtype=np.float64) for c in mtz_m.columns}
                mI = mcols.get("I", mcols.get("I(+)", np.array([])))
                mS = mcols.get("SigI", mcols.get("SigI(+)", np.array([])))
                val_m = np.isfinite(mI) & np.isfinite(mS) & (mS > 0)
                if np.any(val_m):
                    hm_v = mcols["H"][val_m].astype(int)
                    km_v = mcols["K"][val_m].astype(int)
                    lm_v = mcols["L"][val_m].astype(int)
                    isigi_v = mI[val_m] / mS[val_m]
                    frac_mat_m = np.array(mtz_m.cell.frac.mat, dtype=np.float64)
                    mhkl_v = np.column_stack((hm_v, km_v, lm_v)).astype(np.float64)
                    inv_d2_m = np.maximum(
                        np.sum((mhkl_v @ frac_mat_m) ** 2, axis=1), 1e-6
                    )
                    d_m = 1.0 / np.sqrt(inv_d2_m)
                    edges = np.linspace(
                        float(inv_d2_m.min()), float(inv_d2_m.max()), 11
                    )
                    dmin_obs = float(np.min(d_m))
                    sg_m = mtz_m.spacegroup or gemmi.find_spacegroup_by_name("P 1")
                    asu_m = gemmi.ReciprocalAsu(sg_m)
                    ops_m = sg_m.operations()
                    hmax = int(np.ceil(mtz_m.cell.a / max(dmin_obs, 0.5))) + 1
                    kmax = int(np.ceil(mtz_m.cell.b / max(dmin_obs, 0.5))) + 1
                    lmax = int(np.ceil(mtz_m.cell.c / max(dmin_obs, 0.5))) + 1
                    theo_set: set[tuple[int, int, int]] = set()
                    for hh in range(-hmax, hmax + 1):
                        for kk in range(-kmax, kmax + 1):
                            for ll in range(-lmax, lmax + 1):
                                if hh == 0 and kk == 0 and ll == 0:
                                    continue
                                if ops_m.is_systematically_absent([hh, kk, ll]):
                                    continue
                                d_val = mtz_m.cell.calculate_d([hh, kk, ll])
                                if d_val >= dmin_obs - 1e-5:
                                    hasu, _ = asu_m.to_asu([hh, kk, ll], ops_m)
                                    theo_set.add(
                                        (int(hasu[0]), int(hasu[1]), int(hasu[2]))
                                    )
                    inv_d2_t = (
                        np.maximum(
                            np.sum(
                                (
                                    np.array(list(theo_set), dtype=np.float64)
                                    @ frac_mat_m
                                )
                                ** 2,
                                axis=1,
                            ),
                            1e-6,
                        )
                        if theo_set
                        else np.array([], dtype=np.float64)
                    )

                    inv_d2_p = np.array([], dtype=np.float64)
                    if os.path.isfile(pred_mtz_path):
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            pmtz_b = gemmi.read_mtz_file(pred_mtz_path)
                            pcols_b = {
                                c.label: np.array(c, dtype=np.float64)
                                for c in pmtz_b.columns
                            }
                            if all(k in pcols_b for k in ("H", "K", "L")):
                                phkl_b = np.column_stack(
                                    (pcols_b["H"], pcols_b["K"], pcols_b["L"])
                                )
                                inv_d2_p = np.maximum(
                                    np.sum((phkl_b @ frac_mat_m) ** 2, axis=1),
                                    1e-6,
                                )

                    xv_v0 = np.array([], dtype=np.float64)
                    xv_v1 = np.array([], dtype=np.float64)
                    inv_d2_x = np.array([], dtype=np.float64)
                    if os.path.isfile(xval_mtz_path):
                        xv = gemmi.read_mtz_file(xval_mtz_path)
                        xcols = {
                            c.label: np.array(c, dtype=np.float64) for c in xv.columns
                        }
                        xh = xcols["H"].astype(int)
                        xk = xcols["K"].astype(int)
                        xl = xcols["L"].astype(int)
                        xhalf = xcols.get("half", np.zeros_like(xh)).astype(int)
                        xI = xcols.get("I", xcols.get("F", np.zeros_like(xcols["H"])))
                        xhkl_all = np.column_stack((xh, xk, xl)).astype(np.float64)
                        x_inv_d2 = np.maximum(
                            np.sum((xhkl_all @ frac_mat_m) ** 2, axis=1), 1e-6
                        )
                        m0 = xhalf == 0
                        m1 = xhalf == 1
                        map0 = {
                            (int(hh), int(kk), int(ll)): (float(val), float(id2))
                            for hh, kk, ll, val, id2 in zip(
                                xh[m0], xk[m0], xl[m0], xI[m0], x_inv_d2[m0]
                            )
                            if np.isfinite(val)
                        }
                        map1 = {
                            (int(hh), int(kk), int(ll)): float(val)
                            for hh, kk, ll, val in zip(xh[m1], xk[m1], xl[m1], xI[m1])
                            if np.isfinite(val)
                        }
                        common = [k for k in map0 if k in map1]
                        if common:
                            xv_v0 = np.array(
                                [map0[k][0] for k in common], dtype=np.float64
                            )
                            xv_v1 = np.array(
                                [map1[k] for k in common], dtype=np.float64
                            )
                            inv_d2_x = np.array(
                                [map0[k][1] for k in common], dtype=np.float64
                            )

                    bins = []
                    for b in range(10):
                        hi_edge = edges[b + 1]
                        lo_edge = edges[b]
                        lo_cmp = lo_edge - 1e-12 if b == 0 else lo_edge
                        hi_cmp = hi_edge + 1e-12 if b == 9 else hi_edge
                        bm_m = (inv_d2_m >= lo_cmp) & (
                            inv_d2_m <= hi_cmp if b == 9 else inv_d2_m < hi_cmp
                        )
                        bm_t = (
                            (inv_d2_t >= lo_cmp)
                            & (inv_d2_t <= hi_cmp if b == 9 else inv_d2_t < hi_cmp)
                            if len(inv_d2_t)
                            else np.array([], dtype=bool)
                        )
                        bm_p = (
                            (inv_d2_p >= lo_cmp)
                            & (inv_d2_p <= hi_cmp if b == 9 else inv_d2_p < hi_cmp)
                            if len(inv_d2_p)
                            else np.array([], dtype=bool)
                        )
                        n_obs_b: int | None = (
                            int(np.sum(bm_p)) if len(inv_d2_p) else None
                        )
                        n_m_b = int(np.sum(bm_m))
                        mult_b: float | None = (
                            round(float(n_obs_b) / max(1, n_m_b), 2)
                            if (n_obs_b is not None and n_m_b > 0)
                            else None
                        )
                        n_t_b = max(n_m_b, int(np.sum(bm_t))) if len(bm_t) else n_m_b
                        comp_b = (
                            round(float(100.0 * n_m_b / max(1, n_t_b)), 2)
                            if n_t_b > 0
                            else 0.0
                        )
                        isigi_b = (
                            round(float(np.mean(isigi_v[bm_m])), 2)
                            if n_m_b > 0
                            else 0.0
                        )
                        cc_b: float | None = None
                        if len(inv_d2_x):
                            bm_x = (inv_d2_x >= lo_cmp) & (
                                inv_d2_x <= hi_cmp if b == 9 else inv_d2_x < hi_cmp
                            )
                            if int(np.sum(bm_x)) >= 2:
                                s0 = xv_v0[bm_x]
                                s1 = xv_v1[bm_x]
                                if np.std(s0) > 1e-12 and np.std(s1) > 1e-12:
                                    r_val = float(np.corrcoef(s0, s1)[0, 1])
                                    if np.isfinite(r_val):
                                        cc_b = round(r_val, 4)
                        d_hi = float(1.0 / np.sqrt(max(hi_edge, 1e-6)))
                        d_lo = float(1.0 / np.sqrt(max(lo_edge, 1e-6)))
                        bin_entry: dict[str, Any] = {
                            "bin": b,
                            "d_max": round(d_lo, 2),
                            "d_min": round(d_hi, 2),
                            "n_refl": n_m_b,
                            "n_theo": n_t_b,
                            "completeness_pct": comp_b,
                            "mean_isigi": isigi_b,
                            "cc_half": cc_b,
                        }
                        if n_obs_b is not None:
                            bin_entry["n_obs"] = n_obs_b
                        if mult_b is not None:
                            bin_entry["multiplicity"] = mult_b
                        bins.append(bin_entry)
                    metrics_doc.setdefault(
                        "space_group", str(sg_m.xhm() or sg_m.short_name() or "P 1")
                    )
                    metrics_doc.setdefault(
                        "cell",
                        [
                            round(float(mtz_m.cell.a), 4),
                            round(float(mtz_m.cell.b), 4),
                            round(float(mtz_m.cell.c), 4),
                            round(float(mtz_m.cell.alpha), 3),
                            round(float(mtz_m.cell.beta), 3),
                            round(float(mtz_m.cell.gamma), 3),
                        ],
                    )
                    metrics_doc.setdefault("n_merged_asu", len(hm_v))
                    if len(inv_d2_p) > 0:
                        metrics_doc.setdefault("n_observations", len(inv_d2_p))
                        metrics_doc.setdefault(
                            "mean_multiplicity",
                            round(float(len(inv_d2_p)) / max(1, len(hm_v)), 2),
                        )
                    metrics_doc.setdefault(
                        "mean_isigi", round(float(np.mean(isigi_v)), 2)
                    )
                    metrics_doc.setdefault(
                        "median_isigi", round(float(np.median(isigi_v)), 2)
                    )
                    if theo_set:
                        metrics_doc.setdefault(
                            "completeness_pct",
                            round(
                                float(100.0 * len(hm_v) / max(1, len(theo_set))),
                                2,
                            ),
                        )
                    if (
                        len(xv_v0) >= 2
                        and np.std(xv_v0) > 1e-12
                        and np.std(xv_v1) > 1e-12
                    ):
                        r_all = float(np.corrcoef(xv_v0, xv_v1)[0, 1])
                        if np.isfinite(r_all):
                            metrics_doc.setdefault("cc_half", round(r_all, 4))
                    metrics_doc["resolution_bins"] = bins
                    metrics_raw = json.dumps(metrics_doc, indent=2)
                    with open(metrics_path, "w", encoding="utf-8") as fh_out:
                        fh_out.write(metrics_raw)

        if metrics_doc:
            job_host = str(job_record.get("runner_host") or "").strip()
            if job_host and not metrics_doc.get("runner_host"):
                metrics_doc["runner_host"] = job_host
                metrics_raw = json.dumps(metrics_doc, indent=2)
                with (
                    contextlib.suppress(OSError),
                    open(metrics_path, "w", encoding="utf-8") as fh_out,
                ):
                    fh_out.write(metrics_raw)
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                from subhkl.viz.rainbow_zones import (
                    render_merge_table_csv,
                    render_merge_table_html,
                )

                with open(
                    os.path.join(work_dir, "merge_table.csv"),
                    "w",
                    encoding="utf-8",
                ) as fh_csv:
                    fh_csv.write(render_merge_table_csv(metrics_doc))
                with open(
                    os.path.join(work_dir, "merge_table.html"),
                    "w",
                    encoding="utf-8",
                ) as fh_html:
                    fh_html.write(render_merge_table_html(metrics_doc))

        with h5py.File(integrated_h5_path, "r+") as hf:
            cgrp = hf.require_group("careless")
            if "metrics_json" in cgrp:
                del cgrp["metrics_json"]
            cgrp.create_dataset(
                "metrics_json",
                data=np.bytes_(metrics_raw.encode("utf-8")),
            )

            merged_map: dict[tuple[int, int, int], tuple[float, float]] = {}
            if os.path.isfile(merged_mtz_path):
                mtz_m = gemmi.read_mtz_file(merged_mtz_path)
                mcols = {c.label: np.array(c, dtype=np.float64) for c in mtz_m.columns}
                mI = mcols.get("I", mcols.get("I(+)", np.zeros(mtz_m.nreflections)))
                mS = mcols.get(
                    "SIGI",
                    mcols.get(
                        "SigI",
                        mcols.get(
                            "SIGI(+)",
                            mcols.get("SigI(+)", np.ones(mtz_m.nreflections)),
                        ),
                    ),
                )
                for j in range(mtz_m.nreflections):
                    mkey = (
                        int(mcols["H"][j]),
                        int(mcols["K"][j]),
                        int(mcols["L"][j]),
                    )
                    merged_map[mkey] = (float(mI[j]), float(mS[j]))
                if merged_map:
                    asu_keys_arr = np.asarray(list(merged_map.keys()), dtype=np.int32)
                    asu_i_arr = np.asarray(
                        [v[0] for v in merged_map.values()], dtype=np.float64
                    )
                    asu_sigi_arr = np.asarray(
                        [v[1] for v in merged_map.values()], dtype=np.float64
                    )
                    for c_ds_name, c_arr in (
                        ("asu_hkl", asu_keys_arr),
                        ("merged_i", asu_i_arr),
                        ("merged_sigi", asu_sigi_arr),
                    ):
                        if c_ds_name in cgrp:
                            del cgrp[c_ds_name]
                        cgrp.create_dataset(c_ds_name, data=c_arr)

            if "peaks" in hf and os.path.isfile(pred_mtz_path):
                pgrp = hf["peaks"]
                ph: Any = None
                pk: Any = None
                pl: Any = None
                if all(k in pgrp for k in ("h", "k", "l")):
                    ph = np.asarray(pgrp["h"][()], dtype=int)
                    pk = np.asarray(pgrp["k"][()], dtype=int)
                    pl = np.asarray(pgrp["l"][()], dtype=int)
                elif "hkl" in pgrp:
                    hkl_mat = np.asarray(pgrp["hkl"][()], dtype=int)
                    if hkl_mat.ndim == 2 and hkl_mat.shape[1] >= 3:
                        ph, pk, pl = hkl_mat[:, 0], hkl_mat[:, 1], hkl_mat[:, 2]
                if ph is not None and "intensity" in pgrp and "sigma" in pgrp:
                    pI = np.asarray(pgrp["intensity"][()], dtype=float)
                    pS = np.asarray(pgrp["sigma"][()], dtype=float)
                    n_peaks = len(ph)

                    mtz_pred = gemmi.read_mtz_file(pred_mtz_path)
                    sg = mtz_pred.spacegroup or gemmi.find_spacegroup_by_name("P 1")
                    asu = gemmi.ReciprocalAsu(sg)
                    ops = sg.operations()
                    pcols = {
                        c.label: np.array(c, dtype=np.float64) for c in mtz_pred.columns
                    }
                    pred_map: dict[
                        tuple[int, int, int],
                        list[tuple[float, float, float, float, float]],
                    ] = {}
                    # Pull the columns out once.  Looking them up per row with an
                    # array-valued default allocated five full-length arrays on
                    # every iteration: quadratic, ~3 ms per row, most of an hour
                    # for a million observations.
                    n_pred = mtz_pred.nreflections
                    pred_cols = [
                        pcols.get(name, np.full(n_pred, default)).tolist()
                        for name, default in (
                            ("Iobs", 0.0),
                            ("SigIobs", 1.0),
                            ("Ipred", 0.0),
                            ("SigIpred", 1.0),
                            ("Scale", 1.0),
                        )
                    ]
                    pred_keys = zip(
                        pcols["H"].astype(int).tolist(),
                        pcols["K"].astype(int).tolist(),
                        pcols["L"].astype(int).tolist(),
                    )
                    for key, row in zip(pred_keys, zip(*pred_cols)):
                        pred_map.setdefault(key, []).append(row)

                    residuals = np.full(n_peaks, np.nan, dtype=np.float64)
                    scales = np.full(n_peaks, np.nan, dtype=np.float64)
                    ipreds = np.full(n_peaks, np.nan, dtype=np.float64)
                    merged_i = np.full(n_peaks, np.nan, dtype=np.float64)
                    merged_sigi = np.full(n_peaks, np.nan, dtype=np.float64)
                    asu_hkl_arr = np.zeros((n_peaks, 3), dtype=np.int32)

                    for i in range(n_peaks):
                        h_asu, _ = asu.to_asu([int(ph[i]), int(pk[i]), int(pl[i])], ops)
                        key = (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))
                        asu_hkl_arr[i] = key
                        if key in merged_map:
                            merged_i[i], merged_sigi[i] = merged_map[key]
                        cands = pred_map.get(key)
                        if cands:
                            best = min(
                                cands,
                                key=lambda c: abs(c[0] - pI[i]) + abs(c[1] - pS[i]),
                            )
                            iobs, sigiobs, ipred, sigipred, scale = best
                            sig_o = (
                                float(sigiobs)
                                if np.isfinite(sigiobs) and sigiobs > 0.0
                                else 0.0
                            )
                            sig_p = (
                                float(sigipred)
                                if np.isfinite(sigipred) and sigipred > 0.0
                                else 0.0
                            )
                            denom = float(np.hypot(sig_o, 0.5 * sig_p))
                            if (
                                denom > 1e-6
                                and np.isfinite(iobs)
                                and np.isfinite(ipred)
                            ):
                                residuals[i] = (iobs - ipred) / denom
                            scales[i] = scale
                            ipreds[i] = ipred

                    for ds_name, arr in (
                        ("careless_residual", residuals),
                        ("careless_scale", scales),
                        ("careless_ipred", ipreds),
                        ("merged_i", merged_i),
                        ("merged_sigi", merged_sigi),
                        ("asu_hkl", asu_hkl_arr),
                    ):
                        if ds_name in pgrp:
                            del pgrp[ds_name]
                        pgrp.create_dataset(ds_name, data=arr)

        if metrics_doc:
            cc_val = metrics_doc.get("cc_half")
            cc_str = f"{cc_val:.4f}" if isinstance(cc_val, (int, float)) else "N/A"
            comp_val = metrics_doc.get("completeness_pct")
            comp_str = (
                f"{comp_val:.1f}%" if isinstance(comp_val, (int, float)) else "N/A"
            )
            host_sfx = (
                f" | Host={metrics_doc['runner_host']}"
                if metrics_doc.get("runner_host")
                else ""
            )
            job_record["stdout"] = job_record.get("stdout", "") + (
                f"\n[careless] Merged {metrics_doc.get('n_merged_asu', 0)} ASU reflections "
                f"from {metrics_doc.get('n_observations', 0)} observations | "
                f"CC1/2={cc_str} | <I/sigma>={metrics_doc.get('mean_isigi', 0.0):.2f} | "
                f"Multiplicity={metrics_doc.get('mean_multiplicity', 0.0):.2f}x | "
                f"Completeness={comp_str}{host_sfx}\n"
            )

    def _ensure_galaxy_h5_dataset(
        self,
        store: Any,
        local_path: str,
        job_record: dict[str, Any],
        *,
        force_upload: bool = False,
        file_ext: str = "h5",
    ) -> str:
        """Link an existing 'ok' dataset in the Galaxy history or upload with file_type=file_ext, to_posix_lines=False."""
        import glob as _glob
        import re as _re

        fname = os.path.basename(local_path)
        if not os.path.isfile(local_path):
            raise RuntimeError(
                f"Required input file '{fname}' was not found at {local_path}; "
                "run the previous pipeline stage first or select it from History."
            )
        local_size = os.path.getsize(local_path)
        prov_ds_id, prov_file_size = self._read_h5_galaxy_provenance(local_path)
        gi = store.nova_connection.galaxy_instance
        known_map: dict[str, str] = getattr(self, "_dataset_id_by_path", {})
        if not force_upload:
            cand_paths = [
                os.path.abspath(local_path),
                os.path.realpath(local_path),
            ]
            parent_d = os.path.dirname(os.path.abspath(local_path))
            parent_bn = os.path.basename(parent_d)
            if parent_bn.startswith("subhkl_work_") and (
                fname == "merged.h5" or fname.lower().endswith("_merged.h5")
            ):
                tag_part = parent_bn[len("subhkl_work_") :]
                cand_paths.append(os.path.join(parent_d, f"{tag_part}_merged.h5"))
                cand_paths.append(os.path.join(parent_d, "merged.h5"))
                for gx_f in _glob.glob(
                    os.path.join(
                        tempfile.gettempdir(),
                        f"subhkl_gx_*_{tag_part}_merged.h5",
                    )
                ):
                    cand_paths.append(os.path.abspath(gx_f))

            for cand_p in cand_paths:
                known_id = known_map.get(cand_p)
                if not known_id:
                    m_gx = _re.match(
                        r"^subhkl_gx_([0-9a-fA-F]+)_",
                        os.path.basename(cand_p),
                    )
                    if m_gx:
                        known_id = m_gx.group(1)
                if (
                    not known_id
                    and cand_p == os.path.abspath(local_path)
                    and prov_ds_id
                ):
                    known_id = prov_ds_id
                if known_id:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        k_info = gi.datasets.show_dataset(known_id)
                        k_name = str(k_info.get("name") or "")
                        is_merged_req = fname == "merged.h5" or fname.lower().endswith(
                            "_merged.h5"
                        )
                        is_wrong_stage = is_merged_req and any(
                            w in k_name.lower()
                            for w in ("solve", "integrat", "predict", ".mtz")
                        )
                        if (
                            k_info.get("state") == "ok"
                            and not k_info.get("deleted")
                            and "/" not in k_name
                            and not is_wrong_stage
                        ):
                            known_map[os.path.abspath(local_path)] = known_id
                            self._dataset_id_by_path = known_map
                            self.touch_dataset(known_id)
                            self.attach_dataset_cell_metadata_from_file(
                                known_id,
                                local_path,
                                fallback_params=job_record.get("params"),
                            )
                            job_record["stdout"] += (
                                f"[nova -> galaxy] Linked cached dataset '{fname}' (id={known_id}, state=ok)\n"
                            )
                            return known_id

            search_names = [fname]
            if (
                fname == "merged.h5"
                and parent_bn.startswith("subhkl_work_")
                and len(parent_bn) > len("subhkl_work_")
            ):
                tag_part = parent_bn[len("subhkl_work_") :]
                search_names = [f"{tag_part}_merged.h5", "merged.h5"]

            generic_names = {
                "merged.h5",
                "solve.h5",
                "integrated.h5",
                "predictor.h5",
                "reflections.mtz",
                "merged.mtz",
                "scan.h5",
            }
            in_prog_candidates: list[tuple[str, str]] = []
            all_hist_ds: list[dict[str, Any]] | None = None
            for sname in search_names:
                is_generic = sname.lower() in generic_names
                existing = list(
                    gi.datasets.get_datasets(history_id=store.history_id, name=sname)
                    or []
                )
                if sname in ("solve.h5", "integrated.h5", "merged.mtz"):
                    if all_hist_ds is None:
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            all_hist_ds = list(
                                gi.datasets.get_datasets(history_id=store.history_id)
                                or []
                            )
                    for d_item in all_hist_ds or []:
                        d_nm = str(d_item.get("name") or "")
                        if d_nm.startswith(f"{sname} (") and d_item not in existing:
                            existing.append(d_item)
                existing.sort(key=lambda d: int(d.get("hid") or 0))
                # Pass 1: Newest-first search for any dataset already in state='ok'
                for ds in reversed(existing):
                    ds_id = ds.get("id")
                    if not ds_id:
                        continue
                    info: dict[str, Any] = {}
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        raw_info = gi.datasets.show_dataset(ds_id)
                        if isinstance(raw_info, dict):
                            info = raw_info
                    if info.get("deleted") is True or ds.get("deleted") is True:
                        continue
                    ds_actual_name = str(info.get("name") or ds.get("name") or "")
                    if "/" in ds_actual_name or not (
                        ds_actual_name == sname
                        or ds_actual_name.startswith(f"{sname} (")
                    ):
                        continue
                    st = (
                        info.get("state")
                        if isinstance(info.get("state"), str)
                        else ds.get("state")
                    )
                    ds_sz = int(info.get("file_size") or 0)
                    valid_sizes = {s for s in (local_size, prov_file_size) if s > 0}
                    if st == "ok":
                        if (
                            is_generic
                            and valid_sizes
                            and ds_sz > 0
                            and ds_sz not in valid_sizes
                        ):
                            continue
                        known_map[os.path.abspath(local_path)] = ds_id
                        self._dataset_id_by_path = known_map
                        self.touch_dataset(ds_id)
                        self.attach_dataset_cell_metadata_from_file(
                            ds_id,
                            local_path,
                            fallback_params=job_record.get("params"),
                        )
                        job_record["stdout"] += (
                            f"[nova -> galaxy] Linked existing dataset '{info.get('name') or sname}' (id={ds_id}, state=ok)\n"
                        )
                        return ds_id
                    if st in ("queued", "running", "new", "upload"):
                        in_prog_candidates.append((ds_id, str(st)))

            # Pass 2: If no 'ok' dataset exists yet, briefly check the newest in-progress dataset
            if in_prog_candidates:
                ds_id, st = in_prog_candidates[0]
                job_record["stdout"] += (
                    f"[nova -> galaxy] Waiting up to 15s for in-progress dataset '{fname}' (id={ds_id}, state={st})...\n"
                )
                with contextlib.suppress(Exception):
                    gi.datasets.wait_for_dataset(
                        ds_id, maxwait=15, interval=2.0, check=False
                    )
                    info = gi.datasets.show_dataset(ds_id)
                    ds_sz = int(info.get("file_size") or 0)
                    if (
                        info.get("state") == "ok"
                        and not info.get("deleted")
                        and (local_size <= 0 or ds_sz <= 0 or ds_sz == local_size)
                    ):
                        known_map[os.path.abspath(local_path)] = ds_id
                        self._dataset_id_by_path = known_map
                        return ds_id

        parent_bn_up = os.path.basename(os.path.dirname(os.path.abspath(local_path)))
        upload_name = fname
        if (
            force_upload
            and parent_bn_up.startswith("subhkl_work_")
            and len(parent_bn_up) > len("subhkl_work_")
        ):
            tag_up = parent_bn_up[len("subhkl_work_") :]
            if (
                tag_up
                and tag_up.lower()
                not in (
                    "dataset",
                    "work",
                    "solve",
                    "integrated",
                    "integrator",
                    "predictor",
                    "merged",
                )
                and fname
                in (
                    "solve.h5",
                    "integrated.h5",
                    "predictor.h5",
                    "reflections.mtz",
                    "merged.mtz",
                    "merge_table.csv",
                    "merge_table.html",
                    "careless_metrics.json",
                )
                and not fname.lower().startswith(f"{tag_up.lower()}_")
            ):
                upload_name = f"{tag_up}_{fname}"

        if file_ext == "mtz" and not self._galaxy_has_datatype("mtz"):
            file_ext = "binary"
        is_local_gx = any(
            h in (getattr(gi, "base_url", "") or self.galaxy_url or "")
            for h in ("127.0.0.1", "localhost")
        )
        job_record["stdout"] += (
            f"[nova -> galaxy] Uploading '{upload_name}' to Galaxy history (file_type={file_ext})...\n"
        )
        up_info = gi.tools.upload_file(
            path=local_path,
            history_id=store.history_id,
            file_name=upload_name,
            file_type=file_ext,
            to_posix_lines=False,
            link_to_files=is_local_gx,
        )
        ds_id = up_info["outputs"][0]["id"]
        with contextlib.suppress(Exception):
            gi.datasets.wait_for_dataset(ds_id, maxwait=120, interval=2.0, check=False)
        known_map[os.path.abspath(local_path)] = ds_id
        self._dataset_id_by_path = known_map
        self._dataset_history_by_id[str(ds_id)] = str(store.history_id)
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            safe_name = "".join(
                ch if (ch.isalnum() or ch in "._-") else "_"
                for ch in os.path.basename(upload_name)
            )
            cache_dest = os.path.join(
                _get_staging_root(), f"subhkl_gx_{ds_id}_{safe_name}"
            )
            register_temp_artifact(cache_dest)
            if os.path.isfile(local_path) and not os.path.exists(cache_dest):
                try:
                    os.link(local_path, cache_dest)
                except OSError:
                    shutil.copy2(local_path, cache_dest)
        self.touch_dataset(ds_id)
        self.attach_dataset_cell_metadata_from_file(
            ds_id,
            local_path,
            fallback_params=job_record.get("params"),
        )
        job_record["stdout"] += (
            f"[nova -> galaxy] Saved '{upload_name}' in Galaxy history (id={ds_id})\n"
        )
        return ds_id

    def _ensure_galaxy_shared_scan_pointer(
        self,
        store: Any,
        merged_path: str,
        job_record: dict[str, Any],
        *,
        work_dir: str = "",
        instrument: str = "",
    ) -> str:
        """Ensure ``merged_path`` resides in mandatory ``SUBHKL_SHARED_DIR`` and return a tiny JSON pointer HDA id."""
        from subhkl.io.shared_storage import (
            is_safe_shared_intermediate,
            stage_file_to_shared_dir,
            to_shared_network_path,
            write_shared_pointer,
        )

        self.require_shared_dir_for_galaxy_mode()
        gi = store.nova_connection.galaxy_instance
        real_m = os.path.abspath(os.path.realpath(merged_path))
        if not os.path.isfile(real_m):
            raise FileNotFoundError(f"Merged scan HDF5 not found: {merged_path}")

        parent_bn = os.path.basename(
            os.path.abspath(work_dir or os.path.dirname(merged_path))
        )
        tag_part = (
            parent_bn[len("subhkl_work_") :]
            if parent_bn.startswith("subhkl_work_")
            and len(parent_bn) > len("subhkl_work_")
            else ""
        )

        if is_safe_shared_intermediate(real_m):
            shared_local = real_m
            net_path = to_shared_network_path(shared_local)
        else:
            # Write-once to a brand-new UUIDv4 path in SUBHKL_SHARED_DIR + os.fsync + close BEFORE pointer publish.
            shared_local, net_path, _uid = stage_file_to_shared_dir(
                real_m,
                artifact_name="merged.h5",
            )
            if work_dir and os.path.isdir(work_dir):
                _symlink_shared_into_work_dir(
                    shared_local, os.path.join(work_dir, "merged.h5")
                )

        ptr_map: dict[str, str] = getattr(self, "_shared_pointer_ds_by_path", {})
        known_ptr_id = ptr_map.get(os.path.abspath(shared_local))
        if known_ptr_id:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                info = gi.datasets.show_dataset(known_ptr_id)
                if (
                    isinstance(info, dict)
                    and info.get("state") == "ok"
                    and not info.get("deleted")
                ):
                    register_shared_intermediate(
                        shared_local,
                        network_path=net_path,
                        dataset_id=known_ptr_id,
                        history_id=str(store.history_id or ""),
                        work_dir=work_dir,
                    )
                    self.touch_dataset(known_ptr_id)
                    job_record["stdout"] += (
                        f"[nova -> galaxy] Linked shared scan pointer (id={known_ptr_id}) -> {net_path}\n"
                    )
                    return str(known_ptr_id)

        ptr_label = f"{tag_part}_scan.ref.json" if tag_part else "scan.ref.json"
        ptr_tmp = os.path.join(
            _get_staging_root(),
            f"subhkl_ptr_{os.path.basename(shared_local)}.ref.json",
        )
        register_temp_artifact(ptr_tmp)
        write_shared_pointer(
            ptr_tmp,
            shared_local,
            artifact="merged.h5",
            delete_after_use=False,
            metadata={"instrument": instrument or None, "tag": tag_part or None},
        )
        is_local_gx = any(
            h in (getattr(gi, "base_url", "") or self.galaxy_url or "")
            for h in ("127.0.0.1", "localhost")
        )
        up_info = gi.tools.upload_file(
            path=ptr_tmp,
            history_id=store.history_id,
            file_name=ptr_label,
            file_type="json",
            to_posix_lines=False,
            link_to_files=is_local_gx,
        )
        ds_id = str(up_info["outputs"][0]["id"])
        with contextlib.suppress(Exception):
            gi.datasets.wait_for_dataset(ds_id, maxwait=60, interval=1.5, check=False)
        ptr_map[os.path.abspath(shared_local)] = ds_id
        self._shared_pointer_ds_by_path = ptr_map
        self._dataset_history_by_id[ds_id] = str(store.history_id)
        register_shared_intermediate(
            shared_local,
            network_path=net_path,
            dataset_id=ds_id,
            history_id=str(store.history_id or ""),
            work_dir=work_dir,
        )
        job_record["stdout"] += (
            f"[nova -> galaxy] Uploaded shared scan pointer '{ptr_label}' (id={ds_id}) -> {net_path}\n"
        )
        return ds_id

    def _dispatch_nova_galaxy_thread(
        self,
        stage_id: str,
        params: dict[str, Any],
        job_record: dict[str, Any],
        on_complete_h5: Any = None,
    ) -> None:
        """Background worker that links/uploads datasets, dispatches to Pulsar on IMAGINEX-1, and streams console_output."""
        try:
            if not self._is_local_staging_mode():
                self.require_shared_dir_for_galaxy_mode()
            store = self.get_datastore("IMAGINE-X SubHKL")
            self.history_id = store.history_id
            gi = store.nova_connection.galaxy_instance
            job_record["stdout"] += (
                f"[nova -> galaxy] Attached to Galaxy history_id={store.history_id} on {self.galaxy_url}\n"
            )

            if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
                raw_cell = [
                    params["a"],
                    params["b"],
                    params["c"],
                    params["alpha"],
                    params["beta"],
                    params["gamma"],
                ]
            else:
                raw_cell = params.get("cell") or [
                    params.get("a", 68.7),
                    params.get("b", 69.8),
                    params.get("c", 156.9),
                    params.get("alpha", 90.0),
                    params.get("beta", 90.0),
                    params.get("gamma", 90.0),
                ]
            if "wavelength_min" in params or "wavelength_max" in params:
                wl = [
                    float(params.get("wavelength_min", 2.8)),
                    float(params.get("wavelength_max", 4.5)),
                ]
            else:
                wl = params.get("wavelength_range") or [2.8, 4.5]
            inst = str(params.get("instrument", "CG4D"))
            sg = str(params.get("space_group", "P 21 21 21"))
            merged_path = params.get("images_file") or ""
            merged_has_images = False
            if merged_path and os.path.isfile(str(merged_path)):
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    import h5py

                    with h5py.File(str(merged_path), "r") as hf_chk:
                        merged_has_images = "images" in hf_chk
            if (
                not merged_path
                or not os.path.isfile(merged_path)
                or not merged_has_images
                or _is_readonly_galaxy_dat(merged_path)
            ) and params.get("peaks_file"):
                disc_m = self._discover_or_fetch_galaxy_images_h5(
                    params.get("peaks_file")
                )
                if disc_m:
                    merged_path = disc_m
                elif not merged_has_images:
                    merged_path = ""
            if not merged_path and stage_id not in ("reduce", "merge"):
                raise RuntimeError(
                    "No merged scan HDF5 file is loaded; run Step 1 (Reduce & Merge) "
                    "or select a merged.h5 dataset from History first."
                )
            if merged_path:
                work_dir, merged_path = self._resolve_work_dir_and_merged(merged_path)
            elif params.get("peaks_file") and os.path.isfile(
                str(params.get("peaks_file"))
            ):
                work_dir, _ = self._resolve_work_dir_and_merged(
                    str(params.get("peaks_file"))
                )
                merged_path = ""
            elif params.get("work_dir") and os.path.isdir(str(params.get("work_dir"))):
                work_dir = os.path.abspath(str(params.get("work_dir")))
                merged_path = ""
            else:
                work_dir, merged_path = (_get_staging_root(), "")
            if not params.get("hydrate_only"):
                self.invalidate_subsequent_stages(stage_id, work_dir)
            solve_path = params.get("peaks_file") or os.path.join(work_dir, "solve.h5")
            solve_bn = os.path.basename(solve_path).lower()
            cand_solve = os.path.join(work_dir, "solve.h5")
            if (
                not os.path.isfile(solve_path)
                or solve_bn in ("merged.h5", "integrated.h5", "predictor.h5")
                or solve_bn.endswith("_merged.h5")
            ):
                solve_path = cand_solve
            elif os.path.dirname(os.path.abspath(solve_path)) != os.path.abspath(
                work_dir
            ):
                from subhkl.viz.rainbow_zones import _discover_images_h5

                disc_for_solve = _discover_images_h5(solve_path)
                if not disc_for_solve or os.path.abspath(
                    disc_for_solve
                ) != os.path.abspath(merged_path):
                    solve_path = cand_solve

            tool_inputs: dict[str, Any] = {}
            if stage_id in ("reduce", "solve", "integrate", "merge"):
                required_tool = (
                    "imaginex_subhkl_prepare_peaks"
                    if stage_id == "reduce"
                    else (
                        "imaginex_subhkl_merge"
                        if stage_id == "merge"
                        else "imaginex_subhkl_solve"
                    )
                )
                if not self._has_tool_on_galaxy(required_tool):
                    if (
                        self.galaxy_url
                        and "127.0.0.1" not in self.galaxy_url
                        and "localhost" not in self.galaxy_url
                        and self._is_galaxy_server("http://127.0.0.1:9090")
                    ):
                        from nova.galaxy import Connection

                        job_record["stdout"] += (
                            f"[nova -> galaxy] '{required_tool}' not yet merged on {self.galaxy_url}; "
                            f"switching to local Galaxy instance http://127.0.0.1:9090 (GPU container runner)...\n"
                        )
                        local_conn = Connection(
                            galaxy_url="http://127.0.0.1:9090",
                            galaxy_key="test_key",
                        )
                        store = local_conn.connect().create_data_store(
                            "IMAGINE-X SubHKL"
                        )
                        gi = store.nova_connection.galaxy_instance
                        self.galaxy_url = "http://127.0.0.1:9090"
                        self.api_key = gi.key
                        self.history_id = store.history_id

                    if not self._has_tool_on_galaxy(required_tool):
                        raise RuntimeError(
                            f"subhkl GPU tool '{required_tool}' not yet deployed on {self.galaxy_url} "
                            f"(awaiting MR merge on ndip/galaxy-tools)"
                        )
                    if (
                        stage_id != "reduce"
                        and getattr(self, "_pulsar_queue_stalled_until", 0.0)
                        > time.time()
                    ):
                        raise RuntimeError(
                            f"Galaxy Pulsar queue on {self.galaxy_url} is stalled in state=new"
                        )

            if stage_id == "solve":
                d_min_solve = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
                raw_bin = params.get("binning")
                solve_binning = (
                    int(raw_bin)
                    if raw_bin not in (None, "", "auto", "default")
                    and int(raw_bin) >= 1
                    else None
                )
                refinement_mode = str(params.get("objective", "band"))
                job_record["images_h5_path"] = merged_path
                job_record["solve_params"] = {
                    "instrument": str(inst),
                    "space_group": str(sg),
                    "cell": [float(x) for x in raw_cell[:6]],
                    "wavelength_range": (float(wl[0]), float(wl[1])),
                    "d_min": d_min_solve,
                    "binning": solve_binning,
                }
                primary_tool = "imaginex_subhkl_solve"
                scan_id = self._ensure_galaxy_shared_scan_pointer(
                    store,
                    merged_path,
                    job_record,
                    work_dir=work_dir,
                    instrument=inst,
                )
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "instrument": inst,
                    "spaceGroup": sg,
                    "a": float(raw_cell[0]),
                    "b": float(raw_cell[1]),
                    "c": float(raw_cell[2]),
                    "alpha": float(raw_cell[3]),
                    "beta": float(raw_cell[4]),
                    "gamma": float(raw_cell[5]),
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                    "refinement": refinement_mode,
                    "dMin": d_min_solve,
                }
                if solve_binning is not None:
                    if refinement_mode == "band":
                        tool_inputs["bandBinning"] = solve_binning
                    else:
                        tool_inputs["poissonBinning"] = solve_binning
            elif stage_id == "integrate":
                primary_tool = "imaginex_subhkl_predict"
                scan_id = self._ensure_galaxy_shared_scan_pointer(
                    store,
                    merged_path,
                    job_record,
                    work_dir=work_dir,
                    instrument=inst,
                )
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                d_min_req = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
                tool_inputs = {
                    "scan": {"src": "hda", "id": scan_id},
                    "indexer": {"src": "hda", "id": idx_id},
                    "instrument": inst,
                    "dMin": round(d_min_req, 3),
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                    "space": sg,
                }
            elif stage_id == "reduce":
                raw_nxs = self._resolve_raw_nexus_inputs(params, work_dir)
                if not raw_nxs:
                    raise RuntimeError(
                        f"Using pre-merged scan {os.path.basename(merged_path)} (no raw *.nxs.h5 specified or in {work_dir})"
                    )
                cand_wd, merged_label = self._work_dir_for_reduce(raw_nxs, work_dir)
                if not (
                    params.get("hydrate_only")
                    and params.get("work_dir")
                    and os.path.isdir(str(params.get("work_dir")))
                ):
                    work_dir = cand_wd
                job_record["work_dir"] = work_dir
                job_record["raw_nxs"] = raw_nxs
                job_record["instrument"] = inst
                job_record["merged_label"] = merged_label
                # Route NeXus verification through ONCat OAuth first (instant, no file copy),
                # falling back to NDIP neutrons_load_data only when ONCat is unavailable.
                verified_ids = list(params.get("verified_dataset_ids") or [])
                if not verified_ids and params.get("ingest_mode") in (
                    "oncat",
                    "file",
                    None,
                ):
                    import re as _re

                    first_p = raw_nxs[0]
                    ipts_str = _re.sub(
                        r"^IPTS-",
                        "",
                        str(params.get("ipts") or "").strip(),
                        flags=_re.IGNORECASE,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.IGNORECASE)
                        if m_ip:
                            ipts_str = m_ip.group(1)
                    fac_str = str(
                        params.get("facility")
                        or ("SNS" if first_p.startswith("/SNS/") else "HFIR")
                    )
                    if ipts_str:
                        try:
                            cat = self.browse_oncat_catalog(
                                {
                                    "facility": fac_str,
                                    "instrument": inst,
                                    "ipts": ipts_str,
                                }
                            )
                            run_map = {
                                int(r["run"]): r
                                for r in (cat.get("runs") or [])
                                if r.get("run") is not None
                            }
                            if run_map:
                                missing_r = []
                                for p_chk in raw_nxs:
                                    m_rc = _re.search(
                                        r"_([0-9]+)\.nxs",
                                        os.path.basename(p_chk),
                                        _re.IGNORECASE,
                                    )
                                    if m_rc and int(m_rc.group(1)) not in run_map:
                                        missing_r.append(os.path.basename(p_chk))
                                if missing_r:
                                    raise FileNotFoundError(
                                        f"ONCat IPTS-{ipts_str}: run(s) not found ({', '.join(missing_r[:4])})"
                                    )
                                verified_ids = [
                                    f"oncat:{inst}:IPTS-{ipts_str}:{len(raw_nxs)}"
                                ]
                        except FileNotFoundError:
                            raise
                        except _NOVA_RECOVERABLE_ERRORS:
                            pass

                if verified_ids:
                    job_record["stdout"] += (
                        f"[galaxy -> ndip] Verified dataset(s): {', '.join(str(x) for x in verified_ids)}\n"
                    )
                elif self._has_tool_on_galaxy("neutrons_load_data") and params.get(
                    "ingest_mode"
                ) in ("oncat", "file"):
                    import re as _re

                    nld_inputs: dict[str, Any] = {}
                    first_p = raw_nxs[0]
                    m_run = _re.search(
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.IGNORECASE
                    )
                    ipts_str = _re.sub(
                        r"^IPTS-",
                        "",
                        str(params.get("ipts") or "").strip(),
                        flags=_re.IGNORECASE,
                    )
                    if not ipts_str:
                        m_ip = _re.search(r"/IPTS-([^/]+)/", first_p, _re.IGNORECASE)
                        if m_ip:
                            ipts_str = m_ip.group(1)
                    nld_inputs = self._build_neutrons_load_data_inputs(
                        ingest_mode=str(params.get("ingest_mode") or "oncat"),
                        fac=str(
                            params.get("facility")
                            or ("SNS" if first_p.startswith("/SNS/") else "HFIR")
                        ),
                        inst=inst,
                        ipts_raw=ipts_str,
                        run_num=int(m_run.group(1)) if m_run else None,
                        filepath=first_p,
                    )
                    nld_res = self._run_neutrons_load_data_tool(
                        gi=gi,
                        history_id=store.history_id,
                        nld_inputs=nld_inputs,
                        filepath=first_p,
                    )
                    nld_jid = (nld_res.get("jobs") or [{}])[0].get("id")
                    nld_outs = [
                        o.get("id")
                        for o in (nld_res.get("outputs") or [])
                        if isinstance(o, dict) and o.get("id")
                    ]
                    job_record["stdout"] += (
                        f"[galaxy -> ndip] Dispatched neutrons_load_data ({nld_inputs.get('params|ingest_mode')}, job_id={nld_jid})\n"
                    )
                    for _ in range(18):
                        if not nld_jid:
                            break
                        j_nld = self._api_request(
                            "GET", f"/api/jobs/{nld_jid}?full=true"
                        )
                        st_nld = j_nld.get("state")
                        if st_nld == "ok":
                            break
                        if st_nld in ("error", "deleted"):
                            err_out = (
                                j_nld.get("tool_stdout")
                                or j_nld.get("tool_stderr")
                                or "File not found"
                            ).strip()
                            for oid in nld_outs:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
                            raise FileNotFoundError(
                                f"NDIP neutrons_load_data failed: {err_out.splitlines()[0]}"
                            )
                        time.sleep(1.0)
                if getattr(self, "_pulsar_queue_stalled_until", 0.0) > time.time():
                    raise RuntimeError(
                        f"Galaxy Pulsar queue on {self.galaxy_url} is stalled in state=new"
                    )
                primary_tool = "imaginex_subhkl_prepare_peaks"
                tool_inputs = {
                    "instrument": inst,
                    "a": float(raw_cell[0]),
                    "b": float(raw_cell[1]),
                    "c": float(raw_cell[2]),
                    "alpha": float(raw_cell[3]),
                    "beta": float(raw_cell[4]),
                    "gamma": float(raw_cell[5]),
                    "spaceGroup": sg,
                    "wavelengthMin": float(wl[0]),
                    "wavelengthMax": float(wl[1]),
                }
                for i_nxs, nxs_p in enumerate(raw_nxs):
                    tool_inputs[f"inputs_{i_nxs}|input"] = nxs_p
            elif stage_id == "merge":
                primary_tool = "imaginex_subhkl_merge"
                staging_roots_abs = {
                    _get_staging_root(),
                    os.path.abspath(tempfile.gettempdir()),
                }
                if (
                    work_dir in staging_roots_abs
                    and params.get("peaks_file")
                    and os.path.isfile(str(params.get("peaks_file")))
                ):
                    work_dir, _ = self._resolve_work_dir_and_merged(
                        str(params.get("peaks_file"))
                    )
                mtz_in_path = os.path.join(work_dir, "reflections.mtz")
                if not os.path.isfile(mtz_in_path):
                    self._ensure_reflections_mtz_from_integrated(
                        work_dir,
                        peaks_file=str(params.get("peaks_file") or ""),
                        space_group=str(sg),
                        job_record=job_record,
                    )
                matched_hist_mtz_id: str | None = None
                if not os.path.isfile(mtz_in_path) and self.is_connected:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        hist_items = self.list_history_datasets(limit=100)
                        best_mtz_ds: dict[str, Any] | None = None
                        for ds in hist_items:
                            if ds.get("state") != "ok":
                                continue
                            dname = str(ds.get("name") or "").lower()
                            dext = str(ds.get("extension") or "").lower()
                            is_mtz_cand = dname.endswith(".mtz") or dext == "mtz"
                            if not is_mtz_cand:
                                continue
                            # Skip downstream Careless outputs when searching for unmerged input MTZ
                            if any(
                                dname == out_k or dname.endswith(out_k)
                                for out_k in (
                                    "merged.mtz",
                                    "predictions.mtz",
                                    "xval.mtz",
                                    "careless_0.mtz",
                                    "careless_predictions_0.mtz",
                                    "careless_xval_0.mtz",
                                )
                            ):
                                continue
                            if "reflections.mtz" in dname or "unmerged" in dname:
                                best_mtz_ds = ds
                                break
                            if best_mtz_ds is None:
                                best_mtz_ds = ds
                        if best_mtz_ds is not None:
                            ds_id = str(best_mtz_ds.get("id") or "").strip()
                            if ds_id:
                                dl_p, _ = self.resolve_dataset_to_local_h5(
                                    ds_id, fallback_dir=work_dir
                                )
                                if os.path.isfile(dl_p):
                                    self._link_or_copy_file(dl_p, mtz_in_path)
                                    matched_hist_mtz_id = ds_id
                if not os.path.isfile(mtz_in_path):
                    raise RuntimeError(
                        f"Unmerged reflections.mtz not found in {work_dir}; run Integrate first or upload an .mtz dataset."
                    )
                mtz_id = matched_hist_mtz_id or self._upload_compact_ndip_artifact(
                    store,
                    mtz_in_path,
                    "mtz",
                    job_record=job_record,
                    max_bytes=256 * 1024 * 1024,
                )
                if not mtz_id:
                    mtz_id = self._ensure_galaxy_h5_dataset(
                        store,
                        mtz_in_path,
                        job_record,
                        force_upload=True,
                        file_ext="mtz",
                    )
                merge_mode = self._resolve_careless_merge_mode(
                    params,
                    work_dir=work_dir,
                    merged_path=merged_path,
                    solve_path=solve_path,
                )
                _merge_ver = self._get_tool_version("imaginex_subhkl_merge") or ""
                _legacy_merge_tool = _merge_ver in (
                    "0.1.0",
                    "0.1.0-dev",
                    "0.1.1",
                    "0.1.2",
                )
                if merge_mode == "mono" and _legacy_merge_tool:
                    raise RuntimeError(
                        f"imaginex_subhkl_merge {_merge_ver} on {self.galaxy_url} lacks monochromatic ('mono') mode for TOF data"
                    )
                tool_inputs = {
                    "reflections": {"src": "hda", "id": mtz_id},
                    "mode": merge_mode,
                    "iterations": int(params.get("iterations") or 10000),
                    "studenttDof": float(params.get("studentt_dof") or 4.0),
                    "mergeHalfDatasets": True,
                    "anomalous": bool(params.get("anomalous", False)),
                }
                raw_dmin_gx = params.get("d_min")
                if raw_dmin_gx not in (None, "", "auto", "default"):
                    with contextlib.suppress(TypeError, ValueError):
                        dmin_gx_f = float(raw_dmin_gx)
                        if math.isfinite(dmin_gx_f) and dmin_gx_f > 0.0:
                            tool_inputs["dmin"] = round(dmin_gx_f, 4)
                if not _legacy_merge_tool and _merge_ver:
                    raw_wpb = params.get("wilson_prior_b")
                    if raw_wpb not in (None, "", "auto", "default"):
                        with contextlib.suppress(TypeError, ValueError):
                            wpb_f = float(raw_wpb)
                            if wpb_f > 0.0:
                                tool_inputs["wilsonPriorB"] = wpb_f
                    if "lorentz_correction" in params:
                        tool_inputs["lorentzCorrection"] = bool(
                            params.get("lorentz_correction")
                        )
            else:
                primary_tool = "imaginex_subhkl_metrics"
                idx_id = self._ensure_galaxy_h5_dataset(store, solve_path, job_record)
                tool_inputs = {
                    "peaks": {"src": "hda", "id": idx_id},
                }

            is_two_step_integrate = (
                stage_id == "integrate"
                and self._has_tool_on_galaxy("imaginex_subhkl_integrate")
            )
            is_multi_run_legacy_reduce = (
                stage_id == "reduce"
                and len(job_record.get("raw_nxs") or []) > 1
                and self._get_tool_version("imaginex_subhkl_prepare_peaks").startswith(
                    "0.1"
                )
            )
            if is_multi_run_legacy_reduce:
                import concurrent.futures
                import re as _re

                raw_nxs_all = list(job_record.get("raw_nxs") or [])
                if all(os.path.isfile(p) for p in raw_nxs_all):
                    raise RuntimeError(
                        f"imaginex_subhkl_prepare_peaks 0.1.0-dev on {self.galaxy_url} lacks multi-input batching; "
                        f"switching to multi-core container reduction ({len(raw_nxs_all)} runs)"
                    )
                out_sub = os.path.join(work_dir, "out")
                os.makedirs(out_sub, exist_ok=True)
                # Pre-warm ONCat catalog cache once for the sweep so parallel workers don't re-fetch it per run
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    m_ip = _re.search(
                        r"/IPTS-([0-9]+)/", raw_nxs_all[0], _re.IGNORECASE
                    )
                    if m_ip:
                        fac_pre = (
                            "HFIR"
                            if (
                                raw_nxs_all[0].startswith("/HFIR")
                                or inst.upper().startswith(("CG", "HB"))
                            )
                            else "SNS"
                        )
                        self.browse_oncat_catalog(
                            {
                                "facility": fac_pre,
                                "instrument": inst,
                                "ipts": m_ip.group(1),
                            }
                        )

                job_record["stdout"] += (
                    f"[galaxy -> pulsar] Multi-run reduction ({len(raw_nxs_all)} runs): "
                    f"dispatching parallel per-run {primary_tool} jobs on {self.galaxy_url}...\n"
                )

                def _submit_one_part(
                    arg: tuple[int, str],
                ) -> tuple[int, str, str, list[dict[str, Any]], str]:
                    idx_r, nxs_p = arg
                    bn = os.path.basename(nxs_p).split(".")[0]
                    part_path = os.path.join(out_sub, f"{bn}.reduce.h5")
                    part_inputs = {
                        "instrument": inst,
                        "a": float(raw_cell[0]),
                        "b": float(raw_cell[1]),
                        "c": float(raw_cell[2]),
                        "alpha": float(raw_cell[3]),
                        "beta": float(raw_cell[4]),
                        "gamma": float(raw_cell[5]),
                        "spaceGroup": sg,
                        "wavelengthMin": float(wl[0]),
                        "wavelengthMax": float(wl[1]),
                        "inputs_0|input": nxs_p,
                    }
                    part_res = gi.tools.run_tool(
                        history_id=store.history_id,
                        tool_id=primary_tool,
                        tool_inputs=part_inputs,
                    )
                    p_jid = part_res["jobs"][0]["id"]
                    p_outs = [
                        {
                            "id": d.get("id"),
                            "name": d.get("output_name") or d.get("name"),
                            "extension": d.get("file_ext")
                            or d.get("extension")
                            or "h5",
                        }
                        for d in (part_res.get("outputs") or [])
                        if isinstance(d, dict)
                    ]
                    return idx_r, nxs_p, p_jid, p_outs, part_path

                submit_workers = min(len(raw_nxs_all), 6)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=submit_workers
                ) as sub_pool:
                    submitted_indexed = list(
                        sub_pool.map(_submit_one_part, list(enumerate(raw_nxs_all)))
                    )
                submitted_indexed.sort(key=lambda item: item[0])
                submitted_parts = [
                    (nxs_p, p_jid, p_outs, part_path)
                    for idx_r, nxs_p, p_jid, p_outs, part_path in submitted_indexed
                ]
                for idx_r, (nxs_p, p_jid, _p_outs, _part_path) in enumerate(
                    submitted_parts
                ):
                    job_record["stdout"] += (
                        f"[galaxy -> pulsar] Dispatched Run {idx_r + 1}/{len(raw_nxs_all)} "
                        f"({os.path.basename(nxs_p)}, job_id={p_jid})\n"
                    )

                log_lock = threading.Lock()

                def _poll_one_part(
                    arg: tuple[int, tuple[str, str, list[dict[str, Any]], str]],
                ) -> tuple[int, str, str, str]:
                    idx_r, (nxs_p, p_jid, p_outs, part_path) = arg
                    part_rec: dict[str, Any] = {
                        "stage_id": "reduce",
                        "job_id": p_jid,
                        "outputs": p_outs,
                        "raw_nxs": [nxs_p],
                        "instrument": inst,
                        "merged_label": os.path.basename(part_path),
                        "reduce_part_path": part_path,
                        "state": "running",
                        "stdout": "",
                        "stderr": "",
                    }
                    self._poll_galaxy_job(
                        part_rec,
                        None,
                        final_step=False,
                        work_dir=work_dir,
                    )
                    for po in p_outs:
                        if po.get("id") and self.history_id:
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                self._api_request(
                                    "PUT",
                                    f"/api/histories/{self.history_id}/contents/{po['id']}",
                                    payload={"deleted": True, "purged": True},
                                )
                    st_part = str(part_rec.get("state") or "ok")
                    with log_lock:
                        job_record["stdout"] += (
                            f"[nova <- galaxy] Completed Run {idx_r + 1}/{len(raw_nxs_all)} "
                            f"({os.path.basename(nxs_p)}, state={st_part})\n"
                        )
                    return idx_r, part_path, st_part, str(part_rec.get("stderr") or "")

                poll_workers = min(len(submitted_parts), 8)
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=poll_workers
                ) as poll_pool:
                    polled_results = list(
                        poll_pool.map(_poll_one_part, list(enumerate(submitted_parts)))
                    )
                polled_results.sort(key=lambda item: item[0])
                part_paths: list[str] = []
                for _idx_r, part_path, st_part, err_part in polled_results:
                    if st_part in ("error", "deleted", "paused"):
                        job_record["state"] = st_part
                        if err_part:
                            job_record["stderr"] = err_part
                        return
                    if os.path.isfile(part_path):
                        part_paths.append(part_path)

                job_record.pop("reduce_part_path", None)
                job_record["raw_nxs"] = raw_nxs_all
                if not part_paths:
                    raise RuntimeError(
                        "No reduced HDF5 files were produced by multi-run reduction."
                    )
                merged_label = str(job_record.get("merged_label") or "merged.h5")
                merged_dest = os.path.join(work_dir, merged_label)
                merged_link = os.path.join(work_dir, "merged.h5")
                import h5py
                import numpy as np

                from subhkl.commands import run_merge_images

                job_record["stdout"] += (
                    f"\n[subhkl] Merging {len(part_paths)} reduced runs (LZF compressed) -> {merged_dest}...\n"
                )
                run_merge_images(
                    " ".join(part_paths),
                    merged_dest,
                    float(raw_cell[0]),
                    float(raw_cell[1]),
                    float(raw_cell[2]),
                    float(raw_cell[3]),
                    float(raw_cell[4]),
                    float(raw_cell[5]),
                    sg,
                )
                with (
                    contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS),
                    h5py.File(merged_dest, "r+") as hf,
                ):
                    if "files" in hf:
                        del hf["files"]
                    hf["files"] = np.array(
                        [f"out/{os.path.basename(p)}".encode() for p in part_paths]
                    )
                self._stamp_merged_metadata(merged_dest, job_record)
                if os.path.abspath(merged_dest) != os.path.abspath(merged_link):
                    try:
                        if os.path.exists(merged_link):
                            os.remove(merged_link)
                        os.link(merged_dest, merged_link)
                    except OSError:
                        shutil.copy2(merged_dest, merged_link)
                job_record["output_h5_path"] = merged_dest
                if callable(on_complete_h5):
                    on_complete_h5(merged_dest)
                    job_record["updated_payload_ready"] = True
                job_record["stdout"] += (
                    f"\n[galaxy -> pulsar] Multi-run reduction ({len(part_paths)} runs) finished (state=ok).\n"
                )
                job_record["state"] = "ok"

                def _bg_preupload() -> None:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        bg_rec: dict[str, Any] = {"stdout": ""}
                        if not self._is_local_staging_mode():
                            up_id = self._ensure_galaxy_shared_scan_pointer(
                                store,
                                merged_dest,
                                bg_rec,
                                work_dir=work_dir,
                                instrument=inst,
                            )
                            job_record["outputs"] = [
                                {
                                    "id": up_id,
                                    "name": f"{merged_label}.ref.json",
                                    "extension": "json",
                                }
                            ]
                            return
                        up_id = self._ensure_galaxy_h5_dataset(
                            store,
                            merged_link,
                            bg_rec,
                            force_upload=True,
                            file_ext="h5",
                        )
                        job_record["outputs"] = [
                            {
                                "id": up_id,
                                "name": merged_label,
                                "extension": "h5",
                            }
                        ]

                threading.Thread(target=_bg_preupload, daemon=True).start()
                return

            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                return
            job_record["tool_id"] = primary_tool
            results = gi.tools.run_tool(
                history_id=store.history_id,
                tool_id=primary_tool,
                tool_inputs=tool_inputs,
            )
            job_id = results["jobs"][0]["id"]
            job_record["job_id"] = job_id
            job_record.setdefault("galaxy_job_ids", []).append(job_id)
            job_record["outputs"] = self._normalize_galaxy_job_outputs(
                results.get("outputs") or [], stage_id=stage_id
            )
            step_lbl = "Step 1/2 " if is_two_step_integrate else ""
            job_record["stdout"] += (
                f"[galaxy -> pulsar] Dispatched {step_lbl}{primary_tool} (Galaxy job_id={job_id}) -> Pulsar GPU queue\n"
            )
            self._poll_galaxy_job(
                job_record,
                None if is_two_step_integrate else on_complete_h5,
                final_step=not is_two_step_integrate,
                work_dir=work_dir,
            )

            # Chain Step 2/2: imaginex_subhkl_integrate (integrator + mtz-exporter)
            if (
                is_two_step_integrate
                and not job_record.get("cancel_requested")
                and job_record.get("state") not in ("error", "deleted", "paused")
                and job_record.get("outputs")
            ):
                pred_ds_id = job_record["outputs"][0]["id"]
                try:
                    gi.datasets.wait_for_dataset(
                        pred_ds_id, maxwait=60, interval=1.5, check=True
                    )
                    int_tool = "imaginex_subhkl_integrate"
                    int_inputs = {
                        "scan": {"src": "hda", "id": scan_id},
                        "predicted": {"src": "hda", "id": pred_ds_id},
                        "instrument": inst,
                        "anisotropic": True,
                        "mosaicFit": True,
                        "normalized": True,
                        "snr": 5.0,
                        "matrix": True,
                        "matrixProfile": "gaussian",
                        "createVisualizations": False,
                    }
                    job_record["tool_id"] = int_tool
                    int_results = gi.tools.run_tool(
                        history_id=store.history_id,
                        tool_id=int_tool,
                        tool_inputs=int_inputs,
                    )
                    int_job_id = int_results["jobs"][0]["id"]
                    job_record["job_id"] = int_job_id
                    job_record.setdefault("galaxy_job_ids", []).append(int_job_id)
                    job_record["outputs"] = self._normalize_galaxy_job_outputs(
                        int_results.get("outputs") or [], stage_id="integrate"
                    )
                    job_record["stdout"] += (
                        f"\n[galaxy -> pulsar] Dispatched Step 2/2 {int_tool} (Galaxy job_id={int_job_id}) -> Pulsar GPU queue\n"
                    )
                    self._poll_galaxy_job(
                        job_record,
                        on_complete_h5,
                        final_step=True,
                        work_dir=work_dir,
                    )
                finally:
                    self.cleanup_ephemeral_predictor_intermediate(
                        pred_ds_id,
                        work_dir=work_dir,
                        job_record=job_record,
                    )
        except FileNotFoundError as fnf_exc:
            clean_msg = self._clean_err_msg(fnf_exc)
            job_record["state"] = "error"
            job_record["stderr"] = f"{clean_msg}\n"
            job_record["stdout"] += f"\n[galaxy -> ndip] Error: {clean_msg}\n"
        except _NOVA_RECOVERABLE_ERRORS as exc:
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                return
            import shutil as _shutil

            clean_msg = self._clean_err_msg(exc)
            is_remote_galaxy = bool(
                self.galaxy_url
                and not any(h in self.galaxy_url for h in ("127.0.0.1", "localhost"))
            )
            has_container_rt = bool(_shutil.which("podman") or _shutil.which("docker"))
            if (
                self.is_explicit_galaxy_exec_mode()
                or "SUBHKL_SHARED_DIR" in clean_msg
                or (is_remote_galaxy and not has_container_rt)
            ):
                job_record["state"] = "error"
                job_record["stderr"] = f"{clean_msg}\n"
                job_record["stdout"] += f"\n[nova -> galaxy] Error: {clean_msg}\n"
                return
            from subhkl.io.parser import detect_runner_hostname

            local_host = detect_runner_hostname()
            job_record["stdout"] += (
                f"[nova -> galaxy] {clean_msg}; running GPU container on {local_host}...\n"
            )
            self._submit_local_subhkl_stage(
                stage_id,
                params,
                on_complete_h5,
                fallback_reason=clean_msg,
                existing_job_record=job_record,
            )
        finally:
            if job_record.get("state") in ("ok", "error", "deleted", "paused"):
                self._sync_job_timing(job_record)

    SNS_HFIR_ROOTS: ClassVar[dict[str, str]] = {"SNS": "/SNS", "HFIR": "/HFIR"}

    @staticmethod
    def _resolve_local_podman_image(
        preferred_image: str,
        *,
        podman_bin: str = "podman",
    ) -> tuple[str, str | None]:
        """Resolve a usable Podman base image when ``preferred_image`` is an unbuilt branch tag or offline."""
        if (
            os.environ.get("SUBHKL_CONTAINER_IMAGE", "").strip()
            or os.environ.get("SUBHKL_CONTAINER_TAG", "").strip()
        ):
            return preferred_image, None

        with contextlib.suppress(OSError, subprocess.SubprocessError):
            exists_res = subprocess.run(
                [podman_bin, "image", "exists", preferred_image],
                capture_output=True,
                timeout=3.0,
                check=False,
            )
            if exists_res.returncode == 0:
                return preferred_image, None

        repo, sep, tag = preferred_image.rpartition(":")
        if not sep or "/" in tag:
            repo, tag = preferred_image, "latest"

        local_images: list[str] = []
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            list_res = subprocess.run(
                [
                    podman_bin,
                    "images",
                    "--format",
                    "{{.Repository}}:{{.Tag}}",
                    repo,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if list_res.returncode == 0:
                for raw_line in list_res.stdout.splitlines():
                    img = raw_line.strip()
                    if (
                        img
                        and img.startswith(f"{repo}:")
                        and not img.endswith(":<none>")
                        and img not in local_images
                    ):
                        local_images.append(img)

        local_fallback: str | None = None
        for candidate in (f"{repo}:main", f"{repo}:latest"):
            if candidate in local_images:
                local_fallback = candidate
                break
        if local_fallback is None and local_images:
            local_fallback = local_images[0]

        is_published_tag = bool(
            tag in ("main", "latest")
            or re.match(r"^[vV]?\d+\.\d+", tag)
            or re.match(r"^pr-\d+$", tag)
        )
        if not is_published_tag:
            if local_fallback is not None:
                return (
                    local_fallback,
                    f"{preferred_image} not built locally; using cached {local_fallback} with bind-mounted subhkl",
                )
            return (
                f"{repo}:main",
                f"{preferred_image} is a git branch tag; falling back to {repo}:main with bind-mounted subhkl",
            )

        if local_fallback is not None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                manifest_res = subprocess.run(
                    [podman_bin, "manifest", "inspect", preferred_image],
                    capture_output=True,
                    timeout=4.0,
                    check=False,
                )
                if manifest_res.returncode != 0:
                    return (
                        local_fallback,
                        f"{preferred_image} unreachable in registry; using cached {local_fallback} with bind-mounted subhkl",
                    )

        return preferred_image, None

    @staticmethod
    def _parse_run_range_spec(spec_str: str) -> list[int]:
        """Parse run specifications such as '2625-2638', '2632-2638/2', '2632:2638:2', or '2632..2638..2'."""
        import re as _re

        runs: list[int] = []
        seen: set[int] = set()
        norm = _re.sub(r"\s*(\.\.|[-–:/])\s*", r"\1", str(spec_str or ""))
        for tok in _re.split(r"[,\s]+", norm):
            t = tok.strip()
            if not t:
                continue
            m = _re.match(r"^(\d+)(?:\.\.|[-–:])(\d+)(?:(?:\.\.|[/:])(\d+)?)?$", t)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                step = int(m.group(3)) if m.group(3) else 1
                if step <= 0:
                    continue
                lo, hi = min(a, b), max(a, b)
                for r in range(lo, hi + 1, step):
                    if len(runs) >= 20000:
                        break
                    if r not in seen:
                        seen.add(r)
                        runs.append(r)
            elif t.isdigit():
                r = int(t)
                if r not in seen:
                    seen.add(r)
                    runs.append(r)
        return runs

    @staticmethod
    def _format_run_list_to_spec(run_numbers: list[int]) -> str:
        uniq = sorted({int(n) for n in run_numbers})
        if not uniq:
            return ""
        parts: list[str] = []
        start = prev = uniq[0]
        for cur in uniq[1:]:
            if cur == prev + 1:
                prev = cur
            else:
                if prev == start:
                    parts.append(str(start))
                elif prev == start + 1:
                    parts.append(f"{start},{prev}")
                else:
                    parts.append(f"{start}-{prev}")
                start = prev = cur
        if prev == start:
            parts.append(str(start))
        elif prev == start + 1:
            parts.append(f"{start},{prev}")
        else:
            parts.append(f"{start}-{prev}")
        return ",".join(parts)

    def _facility_mount_root(self, fac: str) -> str:
        key = str(fac or "HFIR").strip().upper()
        return self.SNS_HFIR_ROOTS.get(key, "/HFIR")

    def _has_local_sns_hfir_fs(self) -> bool:
        return any(
            os.path.isdir(root)
            for root in (
                self._facility_mount_root("SNS"),
                self._facility_mount_root("HFIR"),
            )
        )

    @staticmethod
    def _read_local_nexus_summary(path: str) -> dict[str, Any]:
        """Lightweight read of NeXus HDF5 header metadata (title, sample, angles, duration) without touching event/image arrays."""
        import datetime as _dt
        import re as _re

        info: dict[str, Any] = {
            "title": "",
            "sample_name": "",
            "angles": "",
            "duration_s": 0.0,
            "duration_hms": "",
            "total_counts": 0,
            "size_mb": 0.0,
            "created": "",
        }
        with contextlib.suppress(OSError):
            st = os.stat(path)
            info["size_mb"] = round(st.st_size / (1024.0 * 1024.0), 1)
            info["created"] = _dt.datetime.fromtimestamp(
                st.st_mtime, tz=_dt.UTC
            ).strftime("%Y-%m-%d %H:%M")

        def _read_str(h5_obj: Any, ds_path: str) -> str:
            if ds_path not in h5_obj:
                return ""
            raw = h5_obj[ds_path][()]
            if hasattr(raw, "size") and getattr(raw, "size", 0) > 0:
                raw = raw.flat[0]
            if isinstance(raw, (bytes, bytearray)):
                return raw.decode("utf-8", "replace").strip()
            return str(raw).strip()

        with contextlib.suppress(Exception):
            import h5py

            with h5py.File(path, "r") as f:
                title = _read_str(f, "entry/title")
                sample_name = _read_str(f, "entry/sample/name")
                info["title"] = title
                info["sample_name"] = sample_name
                if "entry/duration" in f:
                    dur_raw = f["entry/duration"][()]
                    dur_s = float(
                        dur_raw.flat[0] if hasattr(dur_raw, "flat") else dur_raw
                    )
                    if dur_s > 0:
                        info["duration_s"] = dur_s
                        tot_sec = round(dur_s)
                        hh, rem = divmod(tot_sec, 3600)
                        mm, ss = divmod(rem, 60)
                        info["duration_hms"] = f"{hh:02d}:{mm:02d}:{ss:02d}"
                if "entry/total_counts" in f:
                    tc_raw = f["entry/total_counts"][()]
                    info["total_counts"] = int(
                        float(tc_raw.flat[0] if hasattr(tc_raw, "flat") else tc_raw)
                    )
                angle_parts: list[str] = []
                if "entry/DASlogs" in f:
                    das = f["entry/DASlogs"]
                    for ax_key, label in (
                        ("omega", "omega"),
                        ("BL11B:Mot:omega", "omega"),
                        ("HB2C:Mot:s1", "omega"),
                        ("chi", "chi"),
                        ("BL11B:Mot:chi", "chi"),
                        ("phi", "phi"),
                        ("BL11B:Mot:phi", "phi"),
                    ):
                        if ax_key in das and "average_value" in das[ax_key]:
                            av = das[ax_key]["average_value"][()]
                            val_f = float(av.flat[0] if hasattr(av, "flat") else av)
                            tag = f"{label}={val_f:.1f}"
                            if not any(p.startswith(f"{label}=") for p in angle_parts):
                                angle_parts.append(tag)
                angles = ";".join(angle_parts)
                if not angles and title:
                    ax_m = _re.findall(
                        r"(axis\s*\d+|omega|phi|chi|kappa)\s*=?\s*([-+]?\d+(?:\.\d+)?)",
                        title,
                        flags=_re.IGNORECASE,
                    )
                    if ax_m:
                        angles = ";".join(
                            f"{k.replace(' ', '').lower()}={v}" for k, v in ax_m
                        )
                info["angles"] = angles
        return info

    def _browse_local_sns_hfir_catalog(
        self,
        fac: str,
        inst: str,
        ipts_raw: str,
        *,
        skip_experiments: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Scan mounted ``/SNS/<INST>`` or ``/HFIR/<INST>`` directories when Galaxy / ONCat is unavailable."""
        import datetime as _dt
        import re as _re

        fac_root = self._facility_mount_root(fac)
        inst_dir = os.path.join(fac_root, inst)
        experiments: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        if not os.path.isdir(inst_dir):
            return experiments, runs

        if not skip_experiments:
            with contextlib.suppress(OSError):
                for entry in os.scandir(inst_dir):
                    m_ip = _re.match(r"^IPTS-(\d+)$", entry.name, _re.IGNORECASE)
                    if not m_ip or not entry.is_dir():
                        continue
                    ip_num = m_ip.group(1)
                    nxs_dir = os.path.join(entry.path, "nexus")
                    run_nums: list[int] = []
                    latest_mtime = 0.0
                    if os.path.isdir(nxs_dir) and os.access(nxs_dir, os.R_OK | os.X_OK):
                        with contextlib.suppress(OSError):
                            for nf in os.scandir(nxs_dir):
                                if not nf.name.endswith((".nxs.h5", ".nxs")):
                                    continue
                                m_r = _re.search(
                                    r"_([0-9]+)\.nxs", nf.name, _re.IGNORECASE
                                )
                                if m_r:
                                    run_nums.append(int(m_r.group(1)))
                                with contextlib.suppress(OSError):
                                    latest_mtime = max(latest_mtime, nf.stat().st_mtime)
                    else:
                        continue
                    run_nums_sorted = sorted(set(run_nums))
                    latest_dt = (
                        _dt.datetime.fromtimestamp(latest_mtime, tz=_dt.UTC).strftime(
                            "%Y-%m-%d"
                        )
                        if latest_mtime > 0
                        else ""
                    )
                    experiments.append(
                        {
                            "ipts": ip_num,
                            "id": f"IPTS-{ip_num}",
                            "n_runs": len(run_nums_sorted),
                            "run_ranges": self._format_run_list_to_spec(
                                run_nums_sorted
                            ),
                            "min_run": run_nums_sorted[0] if run_nums_sorted else None,
                            "max_run": run_nums_sorted[-1] if run_nums_sorted else None,
                            "latest_date": latest_dt,
                            "title": "",
                        }
                    )
            experiments.sort(
                key=lambda x: (
                    1 if (x.get("n_runs") or 0) > 0 else 0,
                    int(x["ipts"]) if str(x.get("ipts", "")).isdigit() else 0,
                ),
                reverse=True,
            )

        if ipts_raw:
            nxs_dir = os.path.join(inst_dir, f"IPTS-{ipts_raw}", "nexus")
            if os.path.isdir(nxs_dir) and os.access(nxs_dir, os.R_OK | os.X_OK):
                entries_by_run: list[tuple[int, str]] = []
                with contextlib.suppress(OSError):
                    for nf in os.scandir(nxs_dir):
                        if (
                            not nf.name.endswith((".nxs.h5", ".nxs"))
                            or not nf.is_file()
                        ):
                            continue
                        m_r = _re.search(r"_([0-9]+)\.nxs", nf.name, _re.IGNORECASE)
                        if m_r:
                            entries_by_run.append((int(m_r.group(1)), nf.path))
                entries_by_run.sort(key=lambda t: t[0])
                read_headers = len(entries_by_run) <= 250
                for r_num, r_path in entries_by_run:
                    if read_headers:
                        meta = self._read_local_nexus_summary(r_path)
                    else:
                        meta = {
                            "title": "",
                            "sample_name": "",
                            "angles": "",
                            "duration_s": 0.0,
                            "duration_hms": "",
                            "total_counts": 0,
                            "size_mb": 0.0,
                            "created": "",
                        }
                        with contextlib.suppress(OSError):
                            st = os.stat(r_path)
                            meta["size_mb"] = round(st.st_size / (1024.0 * 1024.0), 1)
                    runs.append(
                        {
                            "run": r_num,
                            "location": r_path,
                            "title": meta["title"],
                            "sample_name": meta["sample_name"],
                            "angles": meta["angles"],
                            "size_mb": meta["size_mb"],
                            "created": meta["created"],
                            "duration_s": meta["duration_s"],
                            "duration_hms": meta["duration_hms"],
                            "total_counts": meta["total_counts"],
                        }
                    )
        return experiments, runs

    def _remap_to_facility_root(self, p_str: str) -> str:
        import re as _re

        m = _re.match(r"^/(SNS|HFIR)(/.*)$", p_str, _re.IGNORECASE)
        if m:
            fac_key = m.group(1).upper()
            root = self._facility_mount_root(fac_key)
            return root.rstrip("/") + m.group(2)
        return p_str

    def check_neutron_paths(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate NeXus / HDF5 file paths via NDIP ``neutrons_load_data`` when Galaxy is available,
        falling back to direct local ``/SNS`` / ``/HFIR`` filesystem inspection when Galaxy is unavailable.
        """
        import glob as _glob
        import re as _re

        fac = str(body.get("facility") or "HFIR").strip()
        inst = str(body.get("instrument") or "CG4D").strip()
        ipts_raw = _re.sub(
            r"^IPTS-", "", str(body.get("ipts") or "").strip(), flags=_re.IGNORECASE
        )
        raw_paths = body.get("paths") or []
        if isinstance(raw_paths, str):
            raw_paths = [
                s.strip() for s in raw_paths.replace("\n", ",").split(",") if s.strip()
            ]

        expanded: list[str] = [str(p).strip() for p in raw_paths if str(p).strip()]
        if not expanded and ipts_raw and body.get("runs"):
            fac_norm = fac.upper()
            inst_norm = inst.upper()
            if inst_norm in ("MANDI", "TOPAZ") and fac_norm == "HFIR":
                fac_norm = "SNS"
            elif inst_norm in ("CG4D", "IMAGINE") and fac_norm == "SNS":
                fac_norm = "HFIR"
            root_dir = self._facility_mount_root(fac_norm).rstrip("/")
            expanded = [
                f"{root_dir}/{inst_norm}/IPTS-{ipts_raw}/nexus/{inst_norm}_{r}.nxs.h5"
                for r in self._parse_run_range_spec(str(body.get("runs") or ""))
            ]
        if not expanded:
            return {
                "ok": False,
                "ndip_checked": False,
                "existing": [],
                "missing": [],
                "ipts_exists": False,
                "suggested_ipts": None,
                "suggested_paths": [],
                "verified_dataset_ids": [],
                "sample_title": "",
                "sample_goniometer": "",
                "message": "No NeXus / HDF5 file paths specified",
            }

        ok = False
        ndip_checked = False
        checked_via = "ndip_unavailable"
        verified_dataset_ids: list[str] = []
        msg = ""
        sample_title = ""
        sample_goniometer = ""
        suggested_ipts: str | None = None
        suggested_paths: list[str] = []
        oncat_existing: list[str] = []
        oncat_missing: list[str] = []
        prefer_local_fs = self._is_local_staging_mode() and (
            self._has_local_sns_hfir_fs()
            or any(os.path.exists(self._remap_to_facility_root(p)) for p in expanded)
        )
        galaxy_unreachable = (not self.is_connected) or prefer_local_fs

        if (
            not prefer_local_fs
            and self.is_connected
            and self._has_tool_on_galaxy("neutrons_load_data")
        ):
            try:
                ndip_checked = True
                checked_via = "ndip_neutrons_load_data"
                m_p0 = _re.match(
                    r"^/(SNS|HFIR)/([^/]+)/IPTS-([^/]+)/", expanded[0], _re.IGNORECASE
                )
                if m_p0 and (
                    str(body.get("ingest_mode") or "oncat") == "file" or not ipts_raw
                ):
                    fac = m_p0.group(1).upper()
                    inst = m_p0.group(2).upper()
                    ipts_raw = m_p0.group(3)
                elif not ipts_raw:
                    m_ip0 = _re.search(r"/IPTS-([^/]+)/", expanded[0], _re.IGNORECASE)
                    if m_ip0:
                        ipts_raw = m_ip0.group(1)
                # Verify via authenticated pyoncat catalog first (instant metadata check, no file fetch)
                if ipts_raw:
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        cat = self.browse_oncat_catalog(
                            {
                                "facility": fac,
                                "instrument": inst,
                                "ipts": ipts_raw,
                                "skip_experiments": True,
                            }
                        )
                        run_map = {
                            int(r["run"]): r
                            for r in (cat.get("runs") or [])
                            if r.get("run") is not None
                        }
                        if run_map:
                            titles_seen: list[str] = []
                            gonio_seen: list[str] = []
                            for p in expanded:
                                m_r = _re.search(
                                    r"_([0-9]+)\.nxs",
                                    os.path.basename(p),
                                    _re.IGNORECASE,
                                )
                                if m_r:
                                    r_num = int(m_r.group(1))
                                    info = run_map.get(r_num)
                                    if info:
                                        oncat_existing.append(info.get("location") or p)
                                        t_str = str(info.get("title") or "").strip()
                                        g_str = str(info.get("angles") or "").strip()
                                        if t_str and t_str not in titles_seen:
                                            titles_seen.append(t_str)
                                        if g_str and g_str not in gonio_seen:
                                            gonio_seen.append(g_str)
                                    else:
                                        oncat_missing.append(p)
                            sample_title = "; ".join(titles_seen[:3])
                            sample_goniometer = "; ".join(gonio_seen[:3])
                            if oncat_existing or oncat_missing:
                                if oncat_missing:
                                    ok = False
                                    msg = (
                                        f"ONCat: {len(oncat_existing)} found, {len(oncat_missing)} missing "
                                        f"({', '.join(os.path.basename(x) for x in oncat_missing[:4])})"
                                    )
                                else:
                                    ok = True
                                    verified_dataset_ids = [
                                        f"oncat:{fac}:{inst}:IPTS-{ipts_raw}:{len(oncat_existing)}"
                                    ]
                                    msg = (
                                        f"Verified {len(oncat_existing)} run(s) via ONCat "
                                        f"({inst} IPTS-{ipts_raw})"
                                    )
                                    if sample_title:
                                        msg += f" — {sample_title[:60]}"
                        elif (cat.get("auth_status") or {}).get("authenticated"):
                            oncat_missing = list(expanded)
                            ok = False
                            msg = f"ONCat: No accessible runs found for {inst} IPTS-{ipts_raw}"

                if not ok and not msg:
                    store = self.get_datastore("IMAGINE-X SubHKL")
                    gi = store.nova_connection.galaxy_instance
                    first_p = oncat_existing[0] if oncat_existing else expanded[0]
                    m_run = _re.search(
                        r"_([0-9]+)\.nxs", os.path.basename(first_p), _re.IGNORECASE
                    )
                    nld_in = self._build_neutrons_load_data_inputs(
                        ingest_mode=str(body.get("ingest_mode") or "oncat"),
                        fac=fac,
                        inst=inst,
                        ipts_raw=ipts_raw,
                        run_num=int(m_run.group(1)) if m_run else None,
                        filepath=first_p,
                    )
                    nld_res = self._run_neutrons_load_data_tool(
                        gi=gi,
                        history_id=store.history_id,
                        nld_inputs=nld_in,
                        filepath=first_p,
                    )
                    jid = (nld_res.get("jobs") or [{}])[0].get("id")
                    out_ids = [
                        o.get("id")
                        for o in (nld_res.get("outputs") or [])
                        if isinstance(o, dict) and o.get("id")
                    ]
                    for _ in range(45):
                        if not jid:
                            break
                        j = self._api_request("GET", f"/api/jobs/{jid}?full=true")
                        st = j.get("state")
                        if st == "ok":
                            ok = True
                            verified_dataset_ids = out_ids or [
                                f"verified:{len(expanded)}"
                            ]
                            msg = (
                                f"Verified {len(expanded)} run(s) via NDIP Load Neutron Data "
                                f"({inst} IPTS-{ipts_raw or '?'})"
                            )
                            if sample_title:
                                msg += f" — {sample_title[:60]}"
                            if len(expanded) > 1 or not ok:
                                for oid in out_ids:
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{store.history_id}/contents/{oid}",
                                            payload={"deleted": True},
                                        )
                            break
                        if st in ("error", "deleted"):
                            err_out = (
                                j.get("tool_stdout") or j.get("tool_stderr") or ""
                            ).strip()
                            first_line = (
                                err_out.splitlines()[0]
                                if err_out
                                else "File or run not found"
                            )
                            msg = f"NDIP Load Neutron Data: {first_line} ({inst} IPTS-{ipts_raw or '?'})"
                            for oid in out_ids:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    self._api_request(
                                        "PUT",
                                        f"/api/histories/{store.history_id}/contents/{oid}",
                                        payload={"deleted": True},
                                    )
                            break
                        time.sleep(1.0)
                    if not ok and not msg:
                        msg = f"NDIP Load Neutron Data check timed out for {first_p}"
            except _NOVA_RECOVERABLE_ERRORS as exc:
                galaxy_unreachable = True
                ndip_checked = False
                msg = f"NDIP Load Neutron Data error: {self._clean_err_msg(exc)}"
        else:
            galaxy_unreachable = True

        # Fallback: when NDIP Galaxy is unconfigured or unreachable, inspect local /SNS and /HFIR directly
        if galaxy_unreachable and not ok:
            if self._has_local_sns_hfir_fs() or any(
                os.path.exists(self._remap_to_facility_root(p)) for p in expanded
            ):
                checked_via = "local_filesystem"
                fs_existing: list[str] = []
                fs_missing: list[str] = []
                titles_local: list[str] = []
                gonio_local: list[str] = []
                for p in expanded:
                    p_loc = self._remap_to_facility_root(p)
                    if any(ch in p_loc for ch in ("*", "?", "[")):
                        matches = sorted(
                            m
                            for m in _glob.glob(p_loc)
                            if os.path.isfile(m) and os.access(m, os.R_OK)
                        )
                        if matches:
                            fs_existing.extend(matches)
                        else:
                            fs_missing.append(p_loc)
                    elif os.path.isfile(p_loc) and os.access(p_loc, os.R_OK):
                        fs_existing.append(p_loc)
                    else:
                        fs_missing.append(p_loc)

                for p_ok in fs_existing[:6]:
                    meta = self._read_local_nexus_summary(p_ok)
                    t_str = str(meta.get("title") or "").strip()
                    g_str = str(meta.get("angles") or "").strip()
                    if t_str and t_str not in titles_local:
                        titles_local.append(t_str)
                    if g_str and g_str not in gonio_local:
                        gonio_local.append(g_str)
                if titles_local:
                    sample_title = "; ".join(titles_local[:3])
                if gonio_local:
                    sample_goniometer = "; ".join(gonio_local[:3])

                oncat_existing = fs_existing
                oncat_missing = fs_missing
                if fs_existing and not fs_missing:
                    ok = True
                    verified_dataset_ids = [
                        f"local_fs:{fac.upper()}:{inst.upper()}:IPTS-{ipts_raw or '?'}:{len(fs_existing)}"
                    ]
                    msg = (
                        f"Verified {len(fs_existing)} file(s) on local /SNS or /HFIR filesystem "
                        f"({inst} IPTS-{ipts_raw or '?'})"
                    )
                    if sample_title:
                        msg += f" — {sample_title[:60]}"
                else:
                    ok = False
                    # If the user entered the wrong IPTS, scan sibling IPTS-*/nexus for the missing run filename
                    if fs_missing:
                        first_miss_bn = os.path.basename(fs_missing[0])
                        fac_guess = (
                            "SNS"
                            if inst.upper() in ("MANDI", "TOPAZ")
                            else (
                                "HFIR"
                                if inst.upper() in ("CG4D", "IMAGINE")
                                else fac.upper()
                            )
                        )
                        inst_root = os.path.join(
                            self._facility_mount_root(fac_guess), inst.upper()
                        )
                        if os.path.isdir(inst_root):
                            with contextlib.suppress(OSError):
                                for ip_ent in os.scandir(inst_root):
                                    m_ip = _re.match(
                                        r"^IPTS-(\d+)$", ip_ent.name, _re.IGNORECASE
                                    )
                                    if not m_ip or not ip_ent.is_dir():
                                        continue
                                    cand_f = os.path.join(
                                        ip_ent.path, "nexus", first_miss_bn
                                    )
                                    if os.path.isfile(cand_f) and os.access(
                                        cand_f, os.R_OK
                                    ):
                                        suggested_ipts = m_ip.group(1)
                                        suggested_paths = [
                                            os.path.join(
                                                ip_ent.path,
                                                "nexus",
                                                os.path.basename(mp),
                                            )
                                            for mp in expanded
                                        ]
                                        break
                    if fs_existing:
                        msg = (
                            f"Local filesystem: {len(fs_existing)} found, {len(fs_missing)} missing "
                            f"({', '.join(os.path.basename(x) for x in fs_missing[:4])})"
                        )
                    else:
                        msg = (
                            f"Local filesystem: file not found or permission denied "
                            f"({', '.join(os.path.basename(x) for x in fs_missing[:3])})"
                        )
                        if suggested_ipts:
                            msg += f" — found in IPTS-{suggested_ipts}"
            elif not msg:
                msg = "NDIP Galaxy connection or local /SNS, /HFIR mount required to verify neutron data paths"

        return {
            "ok": ok,
            "ndip_checked": ndip_checked,
            "checked_via": checked_via,
            "existing": oncat_existing
            if oncat_existing and ok
            else (list(expanded) if ok else []),
            "missing": oncat_missing
            if oncat_missing
            else ([] if ok else list(expanded)),
            "ipts_exists": ok,
            "suggested_ipts": suggested_ipts,
            "suggested_paths": suggested_paths,
            "verified_dataset_ids": verified_dataset_ids,
            "sample_title": sample_title,
            "sample_goniometer": sample_goniometer,
            "message": msg,
        }

    ONCAT_DEFAULT_URL = "https://oncat.ornl.gov"
    # Public human-user device-authorization client ID published by ONCat docs.
    ONCAT_DEFAULT_CLIENT_ID = "eaeb036a-2602-4bb9-8530-0bb5812da7a1"
    ONCAT_DEFAULT_SCOPES: ClassVar[list[str]] = ["api:read", "openid"]

    def _get_oncat_token_path(self) -> str:
        return os.path.expanduser(
            os.environ.get("ONCAT_TOKEN_PATH", "~/.oncat/token.json")
        )

    def _seed_oidc_token_if_available(self) -> None:
        """If an outer NDIP/Galaxy OIDC token or token_endpoint is present in env, seed pyoncat's token store."""
        import pyoncat  # type: ignore[import-untyped]

        token_path = self._get_oncat_token_path()
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        store = pyoncat.FileSystemTokenStore(token_path)
        if store.read_token():
            return

        raw_token = (
            os.environ.get("ONCAT_TOKEN") or os.environ.get("NDIP_OIDC_TOKEN") or ""
        ).strip()
        if not raw_token:
            token_ep = (
                os.environ.get("GALAXY_TOKEN_ENDPOINT")
                or os.environ.get("PULSAR_TOKEN_ENDPOINT")
                or os.environ.get("TOKEN_ENDPOINT")
                or ""
            ).strip()
            if token_ep:
                provider = os.environ.get("OIDC_PROVIDER", "pingfed").strip()
                try:
                    import urllib.request

                    sep = "&" if "?" in token_ep else "?"
                    with urllib.request.urlopen(
                        f"{token_ep}{sep}provider={provider}", timeout=5
                    ) as resp:
                        raw_token = resp.read().decode("utf-8").strip().strip('"')
                except _NOVA_RECOVERABLE_ERRORS:
                    raw_token = ""

        if raw_token:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                if raw_token.startswith("{"):
                    tok_obj = json.loads(raw_token)
                else:
                    tok_obj = {
                        "access_token": raw_token,
                        "token_type": "Bearer",
                    }
                store.write_token(tok_obj)

    def _build_oauth_oncat_client(self, verification_handler: Any = None) -> Any:
        """Build a pyoncat.ONCat client configured with FileSystemTokenStore and OAuth flow."""
        import pyoncat  # type: ignore[import-untyped]

        self._seed_oidc_token_if_available()
        url = os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
        client_id = os.environ.get("ONCAT_CLIENT_ID", self.ONCAT_DEFAULT_CLIENT_ID)
        client_secret = os.environ.get("ONCAT_CLIENT_SECRET")
        username = os.environ.get("ONCAT_USERNAME")
        password = os.environ.get("ONCAT_PASSWORD")
        token_path = self._get_oncat_token_path()
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        store = pyoncat.FileSystemTokenStore(token_path)

        if client_id and client_secret:
            client = pyoncat.ONCat(
                url,
                client_id=client_id,
                client_secret=client_secret,
                flow=pyoncat.CLIENT_CREDENTIALS_FLOW,
                token_getter=store.read_token,
                token_setter=store.write_token,
            )
            if not client.has_stored_token():
                client.login()
            return client

        if client_id and username and password:
            client = pyoncat.ONCat(
                url,
                client_id=client_id,
                flow=pyoncat.RESOURCE_OWNER_CREDENTIALS_FLOW,
                token_getter=store.read_token,
                token_setter=store.write_token,
            )
            if not client.has_stored_token():
                client.login(username, password)
            return client

        return pyoncat.ONCat(
            url,
            client_id=client_id,
            flow=pyoncat.DEVICE_AUTHORIZATION_FLOW,
            scopes=list(self.ONCAT_DEFAULT_SCOPES),
            token_getter=store.read_token,
            token_setter=store.write_token,
            reauth_on_expired=pyoncat.REAUTH_INTERACTION_REQUIRED,
            verification_handler=verification_handler,
        )

    def _get_oncat_client(self) -> tuple[Any, bool]:
        """Return (pyoncat.ONCat client, is_authenticated).

        Uses stored OAuth/OIDC token via pyoncat when available; falls back to
        anonymous pyoncat client for public catalog fields when not signed in.
        """
        url = os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
        oauth_client = None
        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
            oauth_client = self._build_oauth_oncat_client()
            if oauth_client.has_stored_token():
                return oauth_client, True
        try:
            import pyoncat  # type: ignore[import-untyped]

            return pyoncat.ONCat(url), False
        except _NOVA_RECOVERABLE_ERRORS:
            return oauth_client, False

    def _invalidate_oncat_caches(self) -> None:
        self._oncat_exp_cache = {}
        self._oncat_runs_cache = {}

    def get_oncat_auth_status(self) -> dict[str, Any]:
        """Return current pyoncat OAuth session state and any active device-flow challenge."""
        state: dict[str, Any] = getattr(self, "_oncat_auth_state", {})
        try:
            oauth_client = self._build_oauth_oncat_client()
            has_tok = bool(oauth_client.has_stored_token())
        except _NOVA_RECOVERABLE_ERRORS:
            oauth_client = None
            has_tok = False

        if has_tok and oauth_client is not None:
            user_name = state.get("user")
            if not user_name:
                try:
                    u = oauth_client.User.retrieve("me")
                    if isinstance(u, dict):
                        user_name = u.get("name")
                        if not user_name or user_name == "Unknown":
                            user_name = u.get("id")
                    state["user"] = user_name or "ORNL User"
                    self._oncat_auth_state = state
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    import pyoncat  # type: ignore[import-untyped]

                    if isinstance(exc, pyoncat.InteractionRequiredError):
                        with contextlib.suppress(OSError):
                            tp = self._get_oncat_token_path()
                            if os.path.isfile(tp):
                                os.remove(tp)
                        has_tok = False
                    else:
                        user_name = "ORNL User"
            if has_tok:
                return {
                    "ok": True,
                    "authenticated": True,
                    "status": "signed_in",
                    "user": user_name or "ORNL User",
                    "verification_uri": None,
                    "verification_uri_complete": None,
                    "user_code": None,
                    "expires_in": None,
                    "message": f"Signed in to ONCat as {user_name or 'ORNL User'}",
                }

        if state.get("status") == "pending_verification":
            return {
                "ok": True,
                "authenticated": False,
                "status": "pending_verification",
                "user": None,
                "verification_uri": state.get("verification_uri"),
                "verification_uri_complete": state.get("verification_uri_complete"),
                "user_code": state.get("user_code"),
                "expires_in": state.get("expires_in"),
                "message": state.get("message")
                or "Waiting for browser approval via ORNL PingFed...",
            }

        return {
            "ok": True,
            "authenticated": False,
            "status": state.get("status") or "signed_out",
            "user": None,
            "verification_uri": None,
            "verification_uri_complete": None,
            "user_code": None,
            "expires_in": None,
            "message": state.get("message")
            or "Anonymous mode (sign in to ONCat for run titles & sample metadata)",
        }

    def handle_oncat_auth(self, body: dict[str, Any]) -> dict[str, Any]:
        """Start/cancel pyoncat Device Authorization sign-in, accept forwarded OIDC token, or sign out."""
        action = str(body.get("action") or "status").strip().lower()
        if action == "status":
            return self.get_oncat_auth_status()

        if action == "logout":
            cancel_ev = getattr(self, "_oncat_login_cancel_event", None)
            if cancel_ev is not None:
                cancel_ev.set()
            tp = self._get_oncat_token_path()
            with contextlib.suppress(OSError):
                if os.path.isfile(tp):
                    os.remove(tp)
            self._oncat_auth_state = {
                "status": "signed_out",
                "user": None,
                "message": "Signed out of ONCat.",
            }
            self._invalidate_oncat_caches()
            return self.get_oncat_auth_status()

        if action == "cancel":
            cancel_ev = getattr(self, "_oncat_login_cancel_event", None)
            if cancel_ev is not None:
                cancel_ev.set()
            self._oncat_auth_state = {
                "status": "signed_out",
                "user": None,
                "message": "Sign-in cancelled.",
            }
            return self.get_oncat_auth_status()

        if action == "token":
            raw_tok = str(body.get("access_token") or "").strip()
            if not raw_tok:
                return {
                    "ok": False,
                    "authenticated": False,
                    "status": "error",
                    "message": "No access_token provided",
                }
            tp = self._get_oncat_token_path()
            os.makedirs(os.path.dirname(tp), exist_ok=True)
            try:
                import pyoncat  # type: ignore[import-untyped]

                store = pyoncat.FileSystemTokenStore(tp)
                store.write_token({"access_token": raw_tok, "token_type": "Bearer"})
            except ImportError:
                with open(tp, "w", encoding="utf-8") as fh:
                    json.dump({"access_token": raw_tok, "token_type": "Bearer"}, fh)
            self._oncat_auth_state = {"status": "signed_in", "user": None}
            self._invalidate_oncat_caches()
            return self.get_oncat_auth_status()

        if action == "start":
            prev_cancel = getattr(self, "_oncat_login_cancel_event", None)
            if prev_cancel is not None:
                prev_cancel.set()

            cancel_ev = threading.Event()
            ready_ev = threading.Event()
            self._oncat_login_cancel_event = cancel_ev
            state: dict[str, Any] = {
                "status": "pending_verification",
                "user": None,
                "verification_uri": None,
                "verification_uri_complete": None,
                "user_code": None,
                "expires_in": None,
                "message": "Requesting device code from ORNL PingFed...",
            }
            self._oncat_auth_state = state

            def _on_challenge(challenge: Any) -> None:
                uri = getattr(challenge, "verification_uri", None)
                uri_complete = (
                    getattr(challenge, "verification_uri_complete", None) or uri
                )
                code = getattr(challenge, "user_code", None)
                exp = getattr(challenge, "expires_in", None)
                state.update(
                    {
                        "status": "pending_verification",
                        "verification_uri": uri,
                        "verification_uri_complete": uri_complete,
                        "user_code": code,
                        "expires_in": exp,
                        "message": f"Approve in browser (Code: {code})",
                    }
                )
                ready_ev.set()

            def _worker() -> None:
                try:
                    client = self._build_oauth_oncat_client(
                        verification_handler=_on_challenge
                    )
                    client.login(
                        cancel_event=cancel_ev,
                        device_authorization_timeout=300.0,
                    )
                    user_name = "ORNL User"
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        u = client.User.retrieve("me")
                        if isinstance(u, dict):
                            user_name = (
                                u.get("name")
                                if u.get("name") and u.get("name") != "Unknown"
                                else (u.get("id") or "ORNL User")
                            )
                    state.update(
                        {
                            "status": "signed_in",
                            "user": user_name,
                            "verification_uri": None,
                            "verification_uri_complete": None,
                            "user_code": None,
                            "expires_in": None,
                            "message": f"Signed in to ONCat as {user_name}",
                        }
                    )
                    self._invalidate_oncat_caches()
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    if not cancel_ev.is_set():
                        state.update(
                            {
                                "status": "signed_out",
                                "user": None,
                                "verification_uri": None,
                                "verification_uri_complete": None,
                                "user_code": None,
                                "expires_in": None,
                                "message": f"ONCat sign-in ended: {self._clean_err_msg(exc)}",
                            }
                        )
                finally:
                    ready_ev.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            ready_ev.wait(timeout=3.0)
            return self.get_oncat_auth_status()

        return self.get_oncat_auth_status()

    def browse_oncat_catalog(self, params: dict[str, Any]) -> dict[str, Any]:
        """Browse IPTS projects and runs via pyoncat (using OAuth token when signed in)."""
        import re as _re

        fac = str(params.get("facility") or "HFIR").strip().upper()
        inst = str(params.get("instrument") or "CG4D").strip().upper()
        if inst in ("MANDI", "TOPAZ") and fac == "HFIR":
            fac = "SNS"
        elif inst in ("CG4D", "IMAGINE") and fac == "SNS":
            fac = "HFIR"
        ipts_raw = _re.sub(
            r"^IPTS-", "", str(params.get("ipts") or "").strip(), flags=_re.IGNORECASE
        )

        # Multi-run support:
        # NDIP's raw `neutrons_load_data` (oncat mode) accepts 1 integer run per tool invocation,
        # whereas `imaginex_subhkl_prepare_peaks` (and local GPU container reduction) accepts
        # multiple run locations resolved by pyoncat and merges them via `subhkl merge-images`.
        has_prepare_peaks = not self.is_connected or self._has_tool_on_galaxy(
            "imaginex_subhkl_prepare_peaks"
        )
        multi_run_supported = bool(has_prepare_peaks)
        multi_run_note = (
            "Multi-run selection supported via pyoncat + subhkl merge-images"
            if multi_run_supported
            else "Single run only (NDIP neutrons_load_data integer run limit)"
        )

        raw_client = self._get_oncat_client()
        if isinstance(raw_client, tuple):
            oncat, is_auth = raw_client
        else:
            oncat, is_auth = raw_client, False
        auth_status = self.get_oncat_auth_status()
        is_auth = bool(auth_status.get("authenticated") or is_auth)

        exp_cache: dict[tuple[str, str, bool], tuple[float, list[dict[str, Any]]]] = (
            getattr(self, "_oncat_exp_cache", {})
        )
        now = time.time()
        experiments: list[dict[str, Any]] = []
        skip_exps = bool(params.get("skip_experiments")) and bool(ipts_raw)
        cached_exp = exp_cache.get((fac, inst, is_auth))
        if cached_exp and (now - cached_exp[0]) < 300.0:
            experiments = cached_exp[1]
        elif not skip_exps:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                try:
                    raw_exps = oncat.Experiment.list(
                        facility=fac,
                        instrument=inst,
                    )
                except _NOVA_RECOVERABLE_ERRORS:
                    if is_auth:
                        import pyoncat as _pyoncat  # type: ignore[import-untyped]

                        oncat = _pyoncat.ONCat(
                            os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
                        )
                        is_auth = False
                        raw_exps = oncat.Experiment.list(facility=fac, instrument=inst)
                    else:
                        raise

                for e in raw_exps:
                    eid = str(e.get("id") or "")
                    m_ip = _re.search(r"IPTS-(\d+)", eid, _re.IGNORECASE)
                    ip_num = m_ip.group(1) if m_ip else eid
                    n_runs = int(e.get("size") or 0)
                    r_info = (e.get("indexed") or {}).get("run_number") or {}
                    ranges_str = str(r_info.get("ranges") or "")
                    latest_dt = str((e.get("latest") or {}).get("created") or "")[:10]
                    title_str = str(e.get("title") or "").strip()
                    experiments.append(
                        {
                            "ipts": ip_num,
                            "id": f"IPTS-{ip_num}",
                            "n_runs": n_runs,
                            "run_ranges": ranges_str,
                            "min_run": r_info.get("min"),
                            "max_run": r_info.get("max"),
                            "latest_date": latest_dt,
                            "title": title_str,
                        }
                    )
                experiments.sort(
                    key=lambda x: (
                        1 if (x.get("n_runs") or 0) > 0 else 0,
                        int(x["ipts"]) if str(x.get("ipts", "")).isdigit() else 0,
                    ),
                    reverse=True,
                )
                exp_cache[(fac, inst, is_auth)] = (now, experiments)
                self._oncat_exp_cache = exp_cache

        runs: list[dict[str, Any]] = []
        if ipts_raw:
            runs_cache: dict[
                tuple[str, str, str, bool], tuple[float, list[dict[str, Any]]]
            ] = getattr(self, "_oncat_runs_cache", {})
            cached_runs = runs_cache.get((fac, inst, ipts_raw, is_auth))
            if cached_runs and (now - cached_runs[0]) < 300.0:
                runs = cached_runs[1]
            else:
                by_run: dict[int, dict[str, Any]] = {}
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    proj = [
                        "indexed.run_number",
                        "location",
                        "size",
                        "created",
                        "indexed.sample_id",
                        "metadata.entry.title",
                        "metadata.entry.sample.name",
                        "metadata.entry.daslogs.omega.average_value",
                        "metadata.entry.daslogs.chi.average_value",
                        "metadata.entry.daslogs.phi.average_value",
                        "metadata.entry.duration",
                        "metadata.entry.total_counts",
                    ]
                    try:
                        dfs = oncat.Datafile.list(
                            facility=fac,
                            instrument=inst,
                            experiment=f"IPTS-{ipts_raw}",
                            projection=proj,
                            exts=[".nxs.h5"],
                        )
                    except _NOVA_RECOVERABLE_ERRORS:
                        if is_auth:
                            import pyoncat as _pyoncat  # type: ignore[import-untyped]

                            oncat = _pyoncat.ONCat(
                                os.environ.get("ONCAT_URL", self.ONCAT_DEFAULT_URL)
                            )
                            is_auth = False
                            dfs = oncat.Datafile.list(
                                facility=fac,
                                instrument=inst,
                                experiment=f"IPTS-{ipts_raw}",
                                projection=proj,
                                exts=[".nxs.h5"],
                            )
                        else:
                            raise

                    for d in dfs:
                        idx_d = d.get("indexed") or {}
                        r_val = idx_d.get("run_number")
                        if r_val is None:
                            m_r = _re.search(
                                r"_([0-9]+)\.nxs",
                                str(d.get("location") or ""),
                                _re.IGNORECASE,
                            )
                            if m_r:
                                r_val = int(m_r.group(1))
                        if r_val is None:
                            continue
                        r_num = int(r_val)
                        meta_entry = (d.get("metadata") or {}).get("entry") or {}
                        loc = str(
                            d.get("location")
                            or f"/{fac}/{inst}/IPTS-{ipts_raw}/nexus/{inst}_{r_num}.nxs.h5"
                        )
                        title = str(meta_entry.get("title") or "").strip()
                        sample_name = str(
                            (meta_entry.get("sample") or {}).get("name") or ""
                        ).strip()
                        daslogs = meta_entry.get("daslogs") or {}
                        angle_parts: list[str] = []
                        for ax_name in ("omega", "chi", "phi"):
                            ax_obj = daslogs.get(ax_name)
                            if (
                                isinstance(ax_obj, dict)
                                and ax_obj.get("average_value") is not None
                            ):
                                try:
                                    val_f = float(ax_obj["average_value"])
                                    angle_parts.append(f"{ax_name}={val_f:.1f}")
                                except (TypeError, ValueError):
                                    pass
                        angles = ";".join(angle_parts)
                        if not angles and title:
                            ax_m = _re.findall(
                                r"(axis\s*\d+|omega|phi|chi|kappa)\s*=?\s*([-+]?\d+(?:\.\d+)?)",
                                title,
                                flags=_re.IGNORECASE,
                            )
                            if ax_m:
                                angles = ";".join(
                                    f"{k.replace(' ', '').lower()}={v}" for k, v in ax_m
                                )
                        sz_bytes = int(d.get("size") or 0)
                        created_str = str(d.get("created") or "")[:16].replace("T", " ")
                        dur_s = float(meta_entry.get("duration") or 0.0)
                        if dur_s > 0:
                            tot_sec = round(dur_s)
                            hh, rem = divmod(tot_sec, 3600)
                            mm, ss = divmod(rem, 60)
                            dur_hms = f"{hh:02d}:{mm:02d}:{ss:02d}"
                        else:
                            dur_hms = ""
                        by_run[r_num] = {
                            "run": r_num,
                            "location": loc,
                            "title": title,
                            "sample_name": sample_name,
                            "angles": angles,
                            "size_mb": round(sz_bytes / (1024.0 * 1024.0), 1),
                            "created": created_str,
                            "duration_s": dur_s,
                            "duration_hms": dur_hms,
                            "total_counts": int(
                                float(meta_entry.get("total_counts") or 0)
                            ),
                        }

                runs = [by_run[k] for k in sorted(by_run.keys())]
                runs_cache[(fac, inst, ipts_raw, is_auth)] = (now, runs)
                self._oncat_runs_cache = runs_cache

        catalog_source = "oncat"
        if (not experiments and not skip_exps) or (ipts_raw and not runs):
            fs_exps, fs_runs = self._browse_local_sns_hfir_catalog(
                fac,
                inst,
                ipts_raw,
                skip_experiments=bool(experiments or skip_exps),
            )
            if not experiments and fs_exps:
                experiments = fs_exps
                catalog_source = "local_filesystem"
            if ipts_raw and not runs and fs_runs:
                runs = fs_runs
                catalog_source = "local_filesystem"

        return {
            "ok": True,
            "facility": fac,
            "instrument": inst,
            "ipts": ipts_raw,
            "catalog_source": catalog_source,
            "oncat_authenticated": is_auth,
            "oncat_auth": auth_status,
            "multi_run_supported": multi_run_supported,
            "multi_run_note": multi_run_note,
            "experiments": experiments,
            "runs": runs,
        }

    def _resolve_raw_nexus_inputs(
        self, params: dict[str, Any], work_dir: str
    ) -> list[str]:
        """Resolve raw .nxs.h5 paths from explicit nexus_files, galaxy_dataset_id, IPTS+runs, or work_dir."""
        import glob as _glob
        import re as _re

        resolved: list[str] = []
        raw_list = params.get("nexus_files")
        if isinstance(raw_list, str):
            raw_list = [
                s.strip() for s in raw_list.replace("\n", ",").split(",") if s.strip()
            ]
        if isinstance(raw_list, list) and raw_list:
            for item in raw_list:
                p_str = self._remap_to_facility_root(str(item).strip())
                if not p_str:
                    continue
                if any(ch in p_str for ch in ("*", "?", "[")):
                    matches = sorted(_glob.glob(p_str))
                    if matches:
                        resolved.extend(matches)
                        continue
                resolved.append(os.path.abspath(p_str))
            if resolved:
                return resolved

        ipts_raw = _re.sub(
            r"^IPTS-", "", str(params.get("ipts") or "").strip(), flags=_re.IGNORECASE
        )
        runs_spec = str(params.get("runs") or "").strip()
        if ipts_raw and runs_spec:
            inst_norm = str(params.get("instrument") or "CG4D").strip().upper()
            fac_norm = str(params.get("facility") or "HFIR").strip().upper()
            if inst_norm in ("MANDI", "TOPAZ") and fac_norm == "HFIR":
                fac_norm = "SNS"
            elif inst_norm in ("CG4D", "IMAGINE") and fac_norm == "SNS":
                fac_norm = "HFIR"
            root_dir = self._facility_mount_root(fac_norm).rstrip("/")
            nxs_dir = os.path.join(root_dir, inst_norm, f"IPTS-{ipts_raw}", "nexus")
            for r_num in self._parse_run_range_spec(runs_spec):
                cand_h5 = os.path.join(nxs_dir, f"{inst_norm}_{r_num}.nxs.h5")
                cand_nxs = os.path.join(nxs_dir, f"{inst_norm}_{r_num}.nxs")
                if os.path.isfile(cand_nxs) and not os.path.isfile(cand_h5):
                    resolved.append(cand_nxs)
                else:
                    resolved.append(cand_h5)
            if resolved:
                return resolved

        ds_id = params.get("galaxy_dataset_id")
        if ds_id:
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                local_p, _ = self.resolve_dataset_to_local_h5(
                    str(ds_id), fallback_dir=work_dir
                )
                if local_p and os.path.isfile(local_p):
                    return [local_p]

        if work_dir and os.path.isdir(work_dir):
            return sorted(_glob.glob(os.path.join(work_dir, "*.nxs.h5")))
        return []

    def _submit_local_subhkl_stage(
        self,
        stage_id: str,
        params: dict[str, Any],
        on_complete_h5: Any = None,
        fallback_reason: str | None = None,
        existing_job_record: dict[str, Any] | None = None,
        preserve_downstream: bool = False,
    ) -> dict[str, Any]:
        """Run subhkl stage inside the GPU Docker/Podman container on IMAGINEX-1 and stream live stdout/stderr."""
        import shlex
        import shutil
        import subprocess
        import sys

        preserve_downstream = bool(preserve_downstream or params.get("hydrate_only"))
        if all(k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")):
            raw_cell = [
                params["a"],
                params["b"],
                params["c"],
                params["alpha"],
                params["beta"],
                params["gamma"],
            ]
        else:
            raw_cell = params.get("cell") or [
                params.get("a", 68.7),
                params.get("b", 69.8),
                params.get("c", 156.9),
                params.get("alpha", 90.0),
                params.get("beta", 90.0),
                params.get("gamma", 90.0),
            ]
        cell_str = ",".join(str(float(x)) for x in raw_cell[:6])
        if "wavelength_min" in params or "wavelength_max" in params:
            wl = [
                float(params.get("wavelength_min", 2.8)),
                float(params.get("wavelength_max", 4.5)),
            ]
        else:
            wl = params.get("wavelength_range") or [2.8, 4.5]
        inst = params.get("instrument", "CG4D")
        sg = params.get("space_group", "P 21 21 21")
        raw_img = params.get("images_file") or ""
        if (
            not raw_img
            or not os.path.isfile(raw_img)
            or _is_readonly_galaxy_dat(raw_img)
        ) and params.get("peaks_file"):
            from subhkl.viz.rainbow_zones import _discover_images_h5

            disc_m = _discover_images_h5(params.get("peaks_file"))
            if disc_m:
                raw_img = disc_m
        if not raw_img and stage_id not in ("reduce", "merge"):
            raise RuntimeError(
                "No merged scan HDF5 file is loaded; run Step 1 (Reduce & Merge) "
                "or select a merged.h5 dataset from History first."
            )
        staging_root = _get_staging_root()
        if raw_img:
            work_dir, merged_path = self._resolve_work_dir_and_merged(
                os.path.abspath(raw_img)
            )
        elif params.get("peaks_file") and os.path.isfile(str(params.get("peaks_file"))):
            work_dir, merged_path = self._resolve_work_dir_and_merged(
                os.path.abspath(str(params.get("peaks_file")))
            )
        elif params.get("work_dir") and os.path.isdir(str(params.get("work_dir"))):
            work_dir = os.path.abspath(str(params.get("work_dir")))
            merged_path = os.path.join(work_dir, "merged.h5")
        else:
            work_dir, merged_path = self._resolve_work_dir_and_merged(
                os.path.join(staging_root, "merged.h5")
            )
        if stage_id != "reduce" and not preserve_downstream:
            self.invalidate_subsequent_stages(stage_id, work_dir)
        solve_dest = os.path.join(work_dir, "solve.h5")
        pred_dest = os.path.join(work_dir, "predictor.h5")
        integrated_dest = os.path.join(work_dir, "integrated.h5")
        mtz_dest = os.path.join(work_dir, "reflections.mtz")

        solve_path = os.path.abspath(params.get("peaks_file") or solve_dest)
        solve_bn = os.path.basename(solve_path).lower()
        if (
            not os.path.isfile(solve_path)
            or solve_bn in ("merged.h5", "integrated.h5", "predictor.h5")
            or solve_bn.endswith("_merged.h5")
        ):
            solve_path = solve_dest
        elif os.path.dirname(os.path.abspath(solve_path)) != os.path.abspath(work_dir):
            from subhkl.viz.rainbow_zones import _discover_images_h5

            disc_for_solve = _discover_images_h5(solve_path)
            if not disc_for_solve or os.path.abspath(disc_for_solve) != os.path.abspath(
                merged_path
            ):
                solve_path = solve_dest

        ts = int(time.time())
        out_h5 = os.path.join(staging_root, f"subhkl_{stage_id}_{ts}.h5")
        pred_tmp_h5 = os.path.join(staging_root, f"subhkl_pred_{ts}.h5")
        mtz_tmp_path = os.path.join(staging_root, f"subhkl_refl_{ts}.mtz")
        reduce_tmp_dir = os.path.join(staging_root, f"subhkl_reduce_{ts}")
        job_stage_temps = [
            out_h5,
            out_h5[:-3] + ".blind.h5",
            out_h5[:-3] + ".stages.h5",
            out_h5[:-3] + ".blind.rung.tmp.h5",
            out_h5[:-3] + ".blind.final.tmp.h5",
            out_h5[:-3] + ".stages.tmp.h5",
            pred_tmp_h5,
            mtz_tmp_path,
            reduce_tmp_dir,
        ]
        for _tmp_item in job_stage_temps:
            register_temp_artifact(_tmp_item)
        raw_nxs_local: list[str] = []
        merged_label = "merged.h5"
        if stage_id == "reduce":
            resolved_candidates = self._resolve_raw_nexus_inputs(params, work_dir)
            raw_nxs_local = [
                p
                for p in resolved_candidates
                if os.path.isfile(p) and os.access(p, os.R_OK)
            ]
            if resolved_candidates and not raw_nxs_local:
                missing_sample = ", ".join(
                    os.path.basename(p) for p in resolved_candidates[:4]
                )
                raise FileNotFoundError(
                    f"Local NeXus file(s) not found or unreadable on /SNS or /HFIR: {missing_sample}"
                )
            if raw_nxs_local:
                cand_wd, merged_label = self._work_dir_for_reduce(
                    raw_nxs_local, work_dir
                )
                if not (
                    preserve_downstream
                    and params.get("work_dir")
                    and os.path.isdir(str(params.get("work_dir")))
                ):
                    work_dir = cand_wd
                solve_dest = os.path.join(work_dir, "solve.h5")
                pred_dest = os.path.join(work_dir, "predictor.h5")
                integrated_dest = os.path.join(work_dir, "integrated.h5")
                mtz_dest = os.path.join(work_dir, "reflections.mtz")
            if not preserve_downstream:
                self.invalidate_subsequent_stages("reduce", work_dir)

        if stage_id == "reduce" and raw_nxs_local:
            os.makedirs(reduce_tmp_dir, exist_ok=True)
            sitecustomize_path = os.path.join(reduce_tmp_dir, "sitecustomize.py")
            with open(sitecustomize_path, "w", encoding="utf-8") as sc_f:
                sc_f.write(
                    "import re, h5py, numpy as np\n"
                    "try:\n"
                    "    import subhkl.instrument.goniometer as _gonio\n"
                    "    _orig_get_rot = _gonio.get_rotation_data_from_nexus\n"
                    "    _PATS = {\n"
                    "        'phi': re.compile(r'(?:axis0|phi)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'kappa': re.compile(r'(?:axis1|kappa)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'omega': re.compile(r'(?:axis2|omega)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'chi': re.compile(r'(?:chi)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "        'alpha': re.compile(r'(?:alpha)\\s*=\\s*([-+]?[0-9]*\\.?[0-9]+)', re.I),\n"
                    "    }\n"
                    "    _DEF = {'omega': 0.0, 'alpha': 45.0, 'kappa': 0.0, 'phi': 0.0, 'chi': 0.0}\n"
                    "    def _patched_get_rot(filename, instrument):\n"
                    "        settings = _gonio.reduction_settings[instrument]\n"
                    "        with h5py.File(filename, 'r') as f:\n"
                    "            das = f['entry/DASlogs'] if 'entry/DASlogs' in f else {}\n"
                    "            has_all = all(k.split('#')[0] in das and 'average_value' in das[k.split('#')[0]] for k in settings['Goniometer'])\n"
                    "            if has_all:\n"
                    "                return _orig_get_rot(filename, instrument)\n"
                    "            t = ''\n"
                    "            if 'entry/title' in f:\n"
                    "                raw = f['entry/title'][()]\n"
                    "                if isinstance(raw, np.ndarray) and raw.size > 0: raw = raw.flat[0]\n"
                    "                t = raw.decode('utf-8', 'replace') if isinstance(raw, (bytes, np.bytes_)) else str(raw)\n"
                    "            parsed = {k: float(m.group(1)) for k, p in _PATS.items() if (m := p.search(t))}\n"
                    "            axes, angles, names = [], [], []\n"
                    "            for ax_name, ax_spec in settings['Goniometer'].items():\n"
                    "                ln = ax_name.split('#')[0]\n"
                    "                sk = ln.split(':')[-1].lower()\n"
                    "                val = float(das[ln]['average_value'][0]) if (ln in das and 'average_value' in das[ln]) else parsed.get(sk, _DEF.get(sk))\n"
                    "                if val is not None:\n"
                    "                    axes.append(np.array(ax_spec, dtype=float))\n"
                    "                    angles.append(float(val))\n"
                    "                    names.append(ln)\n"
                    "            return axes, angles, names\n"
                    "    _gonio.get_rotation_data_from_nexus = _patched_get_rot\n"
                    "    _orig_cd = h5py.File.create_dataset\n"
                    "    def _lzf_cd(self, name, shape=None, dtype=None, data=None, **kw):\n"
                    "        sh = shape if shape is not None else getattr(data, 'shape', ())\n"
                    "        if name == 'images' and len(sh) >= 3 and sh[0] > 0:\n"
                    "            kw.setdefault('compression', 'lzf')\n"
                    "            kw.setdefault('chunks', (1,) + tuple(sh[1:]))\n"
                    "        return _orig_cd(self, name, shape=shape, dtype=dtype, data=data, **kw)\n"
                    "    h5py.File.create_dataset = _lzf_cd\n"
                    "except Exception:\n"
                    "    pass\n"
                )
            batch_py_path = os.path.join(reduce_tmp_dir, "_run_batch_reduce.py")
            with open(batch_py_path, "w") as bf:
                bf.write(
                    "import os, sys, concurrent.futures, multiprocessing as mp\n"
                    "os.environ.setdefault('JAX_PLATFORMS', 'cpu')\n"
                    "from subhkl.commands import run_reduce\n"
                    "def _w(t):\n"
                    "    run_reduce(t[0], t[1], t[2], wavelength_min=t[3], wavelength_max=t[4])\n"
                    "if __name__ == '__main__':\n"
                    "    inst = sys.argv[1]\n"
                    "    wl_min, wl_max = float(sys.argv[2]), float(sys.argv[3])\n"
                    "    out_dir = sys.argv[4]\n"
                    "    nxs_files = sys.argv[5:]\n"
                    "    tasks = [(p, os.path.join(out_dir, f\"{os.path.basename(p).split('.')[0] or f'run_{i:03d}'}.reduce.h5\"), inst, wl_min, wl_max) for i, p in enumerate(nxs_files)]\n"
                    "    if len(tasks) == 1:\n"
                    "        _w(tasks[0])\n"
                    "    else:\n"
                    "        workers = min(len(tasks), os.cpu_count() or 8, 16)\n"
                    "        print(f'Reducing {len(tasks)} runs in parallel across {workers} CPU cores...')\n"
                    "        ctx = mp.get_context('fork')\n"
                    "        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:\n"
                    "            list(ex.map(_w, tasks))\n"
                )
            reduced_parts_ordered = [
                os.path.join(
                    reduce_tmp_dir,
                    f"{os.path.basename(p).split('.')[0] or f'run_{i:03d}'}.reduce.h5",
                )
                for i, p in enumerate(raw_nxs_local)
            ]
            py_reduce_bin = "python" if shutil.which("podman") else sys.executable
            r_cmd = [
                py_reduce_bin,
                "-u",
                batch_py_path,
                str(inst),
                str(wl[0]),
                str(wl[1]),
                reduce_tmp_dir,
                *raw_nxs_local,
            ]
            merge_cmd = [
                py_reduce_bin,
                "-u",
                "-m",
                "subhkl.io.parser",
                "merge-images",
                " ".join(reduced_parts_ordered),
                out_h5,
                str(float(raw_cell[0])),
                str(float(raw_cell[1])),
                str(float(raw_cell[2])),
                str(float(raw_cell[3])),
                str(float(raw_cell[4])),
                str(float(raw_cell[5])),
                str(sg),
            ]
            reduce_cmds = [
                shlex.join(r_cmd),
                shlex.join(merge_cmd),
                f"rm -f {shlex.quote(reduce_tmp_dir)}/*.reduce.h5",
            ]
            chained_reduce = (
                f"set -e; export PYTHONPATH={shlex.quote(reduce_tmp_dir)}:${{PYTHONPATH:-}}; "
                + " && ".join(reduce_cmds)
            )
            inner_cmd = ["bash", "-c", chained_reduce]
        elif stage_id == "reduce" and os.path.isfile(merged_path):
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "sum-images",
                out_h5,
                merged_path,
            ]
        elif stage_id == "solve" and os.path.isfile(merged_path):
            from subhkl.viz.rainbow_zones import default_rel_border_for_instrument

            refine_modes_str = str(
                params.get(
                    "refine_modes",
                    "cylindrical,axial_stretch,area,global_rot,global_trans,sample,cell,per_run,per_run_trans",
                )
            )
            rel_border_solve = float(
                params.get(
                    "rel_border_width",
                    default_rel_border_for_instrument(str(inst)),
                )
            )
            solve_bw_px = 4
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py as _h5

                with _h5.File(merged_path, "r") as _mf:
                    if "images/ims" in _mf and _mf["images/ims"].ndim >= 3:
                        _im_sh = _mf["images/ims"].shape
                        solve_bw_px = max(
                            0,
                            round(
                                rel_border_solve * min(int(_im_sh[1]), int(_im_sh[2]))
                            ),
                        )
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "solve",
                merged_path,
                out_h5,
                "--instrument",
                str(inst),
                "--cell",
                cell_str,
                "--space-group",
                str(sg),
                "--wavelength-min",
                str(wl[0]),
                "--wavelength-max",
                str(wl[1]),
                "--d-min",
                str(params.get("d_min") or round(0.5 * float(wl[0]), 2)),
                "--border-width",
                str(solve_bw_px),
                "--refinement",
                str(params.get("objective", "band")),
            ]
            if str(params.get("objective", "band")) == "band" and refine_modes_str:
                inner_cmd.extend(["--refine-modes", refine_modes_str])
            raw_bin_local = params.get("binning")
            if (
                raw_bin_local not in (None, "", "auto", "default")
                and int(raw_bin_local) >= 1
            ):
                inner_cmd.extend(["--binning", str(int(raw_bin_local))])
        elif (
            stage_id == "integrate"
            and os.path.isfile(merged_path)
            and os.path.isfile(solve_path)
        ):
            from subhkl.viz.rainbow_zones import default_rel_border_for_instrument

            d_min_req = float(params.get("d_min") or round(0.5 * float(wl[0]), 2))
            d_min_val = str(round(d_min_req, 3))
            rel_border = float(
                params.get(
                    "rel_border_width",
                    default_rel_border_for_instrument(str(inst)),
                )
            )
            shape_min_snr = float(params.get("shape_fit_min_snr", 5.0))
            has_solve_mask = False
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                import h5py as _h5

                with _h5.File(solve_path, "r") as _sf:
                    has_solve_mask = (
                        "detector/pixel_mask" in _sf and "detector/mask_bank_ids" in _sf
                    )
            pred_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "peak-predictor",
                merged_path,
                str(inst),
                solve_path,
                pred_tmp_h5,
                "--d-min",
                d_min_val,
                "--wavel-min",
                str(wl[0]),
                "--wavel-max",
                str(wl[1]),
                "--space-group",
                str(sg),
                "--rel-border-width",
                str(round(rel_border, 4)),
            ]
            if has_solve_mask:
                pred_cmd.extend(["--static-mask-file", solve_path])
            int_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "integrator",
                merged_path,
                str(inst),
                pred_tmp_h5,
                out_h5,
                "--anisotropic",
                "--fit-mosaicity",
                "--shape-fit-normalized",
                "--shape-fit-min-snr",
                str(round(shape_min_snr, 2)),
                "--allow-negative",
                "--matrix-free",
                "--rel-border-width",
                str(round(rel_border, 4)),
                "--show-progress",
            ]
            if has_solve_mask:
                int_cmd.extend(["--static-mask-file", solve_path])
            mtz_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "mtz-exporter",
                out_h5,
                mtz_tmp_path,
                "--predictions-file",
                pred_tmp_h5,
                "--corrections-file",
                solve_path,
            ]
            chained = (
                f"set -e; "
                f"{shlex.join(pred_cmd)} && "
                f"{shlex.join(int_cmd)} && "
                f"{shlex.join(mtz_cmd)}"
            )
            inner_cmd = ["bash", "-c", chained]
        elif stage_id == "merge":
            replay_src = getattr(self, "replay_h5", None)
            if not os.path.isfile(mtz_dest):
                alt_mtz = (
                    os.path.join(
                        os.path.dirname(os.path.abspath(replay_src)),
                        "reflections.mtz",
                    )
                    if replay_src
                    else ""
                )
                if alt_mtz and os.path.isfile(alt_mtz):
                    shutil.copy2(alt_mtz, mtz_dest)
                else:
                    src_int = (
                        integrated_dest
                        if os.path.isfile(integrated_dest)
                        else (replay_src or "")
                    )
                    if src_int and os.path.isfile(src_int):
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            from subhkl.io.export import MTZExporter

                            exp = MTZExporter(
                                src_int,
                                space_group=str(sg),
                                predictions_file=(
                                    pred_dest if os.path.isfile(pred_dest) else None
                                ),
                                corrections_file=(
                                    solve_path if os.path.isfile(solve_path) else None
                                ),
                            )
                            exp.write_mtz(mtz_dest)
            careless_iters = int(params.get("iterations") or 10000)
            studentt_dof = float(params.get("studentt_dof") or 4.0)
            raw_dmin = params.get("d_min")
            careless_dmin: float | None = None
            if raw_dmin not in (None, "", "auto", "default"):
                with contextlib.suppress(TypeError, ValueError):
                    dmin_cand = float(raw_dmin)
                    if math.isfinite(dmin_cand) and dmin_cand > 0.0:
                        careless_dmin = round(dmin_cand, 4)
            raw_wilson_b = params.get("wilson_prior_b")
            wilson_b = (
                float(raw_wilson_b)
                if raw_wilson_b is not None and str(raw_wilson_b).strip() != ""
                else None
            )
            spectral_file = str(params.get("spectral_file") or "").strip()
            lorentz_corr = bool(params.get("lorentz_correction", bool(spectral_file)))
            merge_mode = self._resolve_careless_merge_mode(
                params,
                work_dir=work_dir,
                merged_path=merged_path,
                solve_path=solve_path,
                integrated_path=integrated_dest,
            )
            merge_work = os.path.join(work_dir, "careless_work")
            os.makedirs(merge_work, exist_ok=True)
            unmerged_mtz = os.path.join(merge_work, "input_unmerged.mtz")
            out_prefix = os.path.join(merge_work, "careless")
            merged_mtz_out = os.path.join(work_dir, "merged.mtz")
            pred_mtz_out = os.path.join(work_dir, "predictions.mtz")
            xval_mtz_out = os.path.join(work_dir, "xval.mtz")
            metrics_json_out = os.path.join(work_dir, "careless_metrics.json")
            prep_py = (
                "import sys, gemmi, numpy as np, reciprocalspaceship as rs\n"
                f"mtz = gemmi.read_mtz_file({mtz_dest!r})\n"
                "c = mtz.cell\n"
                "na = 90.0 if abs(c.alpha - 90) < 0.2 else (120.0 if abs(c.alpha - 120) < 0.2 else c.alpha)\n"
                "nb = 90.0 if abs(c.beta - 90) < 0.2 else (120.0 if abs(c.beta - 120) < 0.2 else c.beta)\n"
                "ng = 90.0 if abs(c.gamma - 90) < 0.2 else (120.0 if abs(c.gamma - 120) < 0.2 else c.gamma)\n"
                "mtz.cell.set(c.a, c.b, c.c, na, nb, ng)\n"
                f"cell_mtz = {os.path.join(merge_work, 'input_cell.mtz')!r}\n"
                "mtz.write_to_file(cell_mtz)\n"
                "ds = rs.read_mtz(cell_mtz)\n"
                "if 'SIGI' in ds:\n"
                "    ds = ds[ds['SIGI'] > 0]\n"
                "if 'BATCH' in ds and 'RUN' not in ds:\n"
                "    ds['RUN'] = rs.DataSeries((ds['BATCH'].to_numpy().astype(int) // 1000).astype(np.float32), index=ds.index, dtype='R')\n"
                "if not ds.get_m_isym_keys():\n"
                "    ds = ds.hkl_to_asu()\n"
                f"_req_dmin = {careless_dmin!r}\n"
                f"_mode = {merge_mode!r}\n"
                "if _req_dmin is not None and len(ds) > 0:\n"
                "    _d_chk = ds.compute_dHKL()['dHKL'].to_numpy(dtype=np.float64)\n"
                "    if _mode == 'poly':\n"
                "        _hkls = ds.get_hkls()\n"
                "        _gcd = np.maximum(1, np.gcd.reduce(np.abs(_hkls), axis=-1))\n"
                "        _d0_max = float(np.nanmax(_d_chk * _gcd)) if len(_d_chk) else 0.0\n"
                "        if _d0_max < float(_req_dmin):\n"
                "            sys.exit(f'No integrated reflections satisfy d_min <= {_req_dmin:.3f} A (max fundamental d = {_d0_max:.3f} A).')\n"
                "    else:\n"
                "        _d_max = float(np.nanmax(_d_chk)) if len(_d_chk) else 0.0\n"
                "        if _d_max < float(_req_dmin):\n"
                "            sys.exit(f'No integrated reflections satisfy d >= {_req_dmin:.3f} A (max d = {_d_max:.3f} A).')\n"
                "ds.merged = False\n"
                f"ds.write_mtz({unmerged_mtz!r})\n"
                "# Restrict neural-scaler metadata strictly to physical beam/detector/run\n"
                "# coordinates -- never Miller indices/asu_id (overfitting) or dHKL\n"
                "# (induces intra-bin resolution-slope correlation).\n"
                "keys = [k for k in ('WAVEL', 'THETA', 'PHI') if k in ds]\n"
                "for extra in ('RUN', 'DPHI', 'DTX', 'DTY', 'DTZ'):\n"
                "    if extra in ds and float(np.nanstd(ds[extra].to_numpy())) > 1e-6:\n"
                "        keys.append(extra)\n"
                "print(','.join(keys))\n"
            )
            post_py = (
                "import json, os, shutil, gemmi, numpy as np\n"
                "import reciprocalspaceship as rs\n"
                "from scipy.stats import pearsonr, spearmanr\n"
                f"shutil.copy2({(out_prefix + '_0.mtz')!r}, {merged_mtz_out!r})\n"
                f"shutil.copy2({(out_prefix + '_predictions_0.mtz')!r}, {pred_mtz_out!r})\n"
                f"if os.path.exists({(out_prefix + '_xval_0.mtz')!r}):\n"
                f"    shutil.copy2({(out_prefix + '_xval_0.mtz')!r}, {xval_mtz_out!r})\n"
                f"summary = {{'status': 'ok', 'mode': {merge_mode!r}}}\n"
                f"if {careless_dmin!r} is not None:\n"
                f"    summary['requested_d_min'] = float({careless_dmin!r})\n"
                f"mtz = gemmi.read_mtz_file({merged_mtz_out!r})\n"
                "cols = {c.label: np.array(c, dtype=np.float64) for c in mtz.columns}\n"
                "I = cols.get('I', cols.get('I(+)', np.array([])))\n"
                "SigI = cols.get('SigI', cols.get('SigI(+)', np.array([])))\n"
                "valid = np.isfinite(I) & np.isfinite(SigI) & (SigI > 0)\n"
                "summary['n_merged_asu'] = int(np.sum(valid))\n"
                "summary['mean_isigi'] = float(np.mean(I[valid] / SigI[valid])) if np.any(valid) else 0.0\n"
                f"pmtz = gemmi.read_mtz_file({pred_mtz_out!r})\n"
                "n_obs = int(pmtz.nreflections)\n"
                "summary['n_observations'] = n_obs\n"
                "summary['mean_multiplicity'] = float(n_obs / max(1, summary['n_merged_asu'])) if summary['n_merged_asu'] > 0 else 0.0\n"
                "try:\n"
                "    h_m = cols['H'][valid].astype(int); k_m = cols['K'][valid].astype(int); l_m = cols['L'][valid].astype(int)\n"
                "    d_m = np.array([mtz.cell.calculate_d([int(hh), int(kk), int(ll)]) for hh, kk, ll in zip(h_m, k_m, l_m)])\n"
                "    dmin_obs = float(np.min(d_m)) if len(d_m) else 2.0\n"
                "    theo_asu = rs.utils.generate_reciprocal_asu(mtz.cell, mtz.spacegroup, dmin_obs)\n"
                "    summary['n_theoretical_asu'] = int(len(theo_asu))\n"
                "    summary['completeness_pct'] = float(100.0 * summary['n_merged_asu'] / max(1, len(theo_asu)))\n"
                "    summary['d_min'] = round(dmin_obs, 3)\n"
                "    summary['d_max'] = round(float(np.max(d_m)), 3) if len(d_m) else 10.0\n"
                "except Exception as e:\n"
                "    summary['completeness_error'] = str(e)\n"
                f"if os.path.exists({xval_mtz_out!r}):\n"
                f"    xv = gemmi.read_mtz_file({xval_mtz_out!r})\n"
                "    xcols = {c.label: np.array(c, dtype=np.float64) for c in xv.columns}\n"
                "    h = xcols['H'].astype(int); k = xcols['K'].astype(int); l = xcols['L'].astype(int)\n"
                "    half = xcols.get('half', np.zeros_like(h)).astype(int)\n"
                "    xI = xcols.get('I', xcols.get('F', np.zeros_like(xcols['H'])))\n"
                "    d_arr = np.array([xv.cell.calculate_d([int(hh), int(kk), int(ll)]) for hh, kk, ll in zip(h, k, l)])\n"
                "    m0 = half == 0; m1 = half == 1\n"
                "    map0 = {(int(hh), int(kk), int(ll)): (float(val), float(dd)) for hh, kk, ll, val, dd in zip(h[m0], k[m0], l[m0], xI[m0], d_arr[m0]) if np.isfinite(val)}\n"
                "    map1 = {(int(hh), int(kk), int(ll)): float(val) for hh, kk, ll, val in zip(h[m1], k[m1], l[m1], xI[m1]) if np.isfinite(val)}\n"
                "    common = [key for key in map0 if key in map1]\n"
                "    if len(common) >= 2:\n"
                "        v0 = np.array([map0[k][0] for k in common]); v1 = np.array([map1[k] for k in common]); d_c = np.array([map0[k][1] for k in common])\n"
                "        summary['cc_half'] = float(pearsonr(v0, v1)[0]) if len(common) >= 3 else float(np.corrcoef(v0, v1)[0, 1])\n"
                "        summary['cc_half_spearman'] = float(spearmanr(v0, v1)[0]) if len(common) >= 3 else summary['cc_half']\n"
                "        inv_d2 = 1.0 / np.maximum(d_c ** 2, 1e-6)\n"
                "        edges = np.linspace(inv_d2.min(), inv_d2.max(), 11)\n"
                "        bins = []\n"
                "        for b in range(10):\n"
                "            bm = (inv_d2 >= edges[b]) & (inv_d2 <= edges[b+1] if b == 9 else inv_d2 < edges[b+1])\n"
                "            cc_b = float(np.corrcoef(v0[bm], v1[bm])[0, 1]) if np.sum(bm) >= 2 else float('nan')\n"
                "            d_hi = float(1.0 / np.sqrt(max(edges[b+1], 1e-6)))\n"
                "            d_lo = float(1.0 / np.sqrt(max(edges[b], 1e-6)))\n"
                "            bins.append({'bin': b, 'd_max': round(d_lo, 2), 'd_min': round(d_hi, 2), 'n_refl': int(np.sum(bm)), 'cc_half': round(cc_b, 4) if np.isfinite(cc_b) else None})\n"
                "        summary['resolution_bins'] = bins\n"
                f"with open({metrics_json_out!r}, 'w') as fh:\n"
                "    json.dump(summary, fh, indent=2)\n"
                f"for _p in ({merged_mtz_out!r}, {pred_mtz_out!r}, {xval_mtz_out!r}, {metrics_json_out!r}):\n"
                "    if os.path.exists(_p):\n"
                "        try: os.chmod(_p, 0o666)\n"
                "        except Exception: pass\n"
                "print('[careless] Merge summary:', json.dumps(summary))\n"
            )
            prep_script = os.path.join(merge_work, "prep_merge.py")
            post_script = os.path.join(merge_work, "post_merge.py")
            with open(prep_script, "w", encoding="utf-8") as fh:
                fh.write(prep_py)
            with open(post_script, "w", encoding="utf-8") as fh:
                fh.write(post_py)
            with contextlib.suppress(OSError):
                os.chmod(prep_script, 0o666)
                os.chmod(post_script, 0o666)
            careless_extra_flags = [
                "--disable-image-scales",
            ]
            if careless_dmin is not None:
                careless_extra_flags.append(f"--dmin={careless_dmin:.6g}")
            if wilson_b is not None:
                careless_extra_flags.append(f"--wilson-prior-b={wilson_b}")
            if spectral_file and os.path.isfile(spectral_file):
                careless_extra_flags.extend(
                    [
                        f"--spectral-file={shlex.quote(spectral_file)}",
                        "--trainable-spectral-scale",
                    ]
                )
                if lorentz_corr:
                    careless_extra_flags.append("--lorentz-correction")
            elif lorentz_corr:
                careless_extra_flags.append("--lorentz-correction")
            extra_flags_str = " ".join(careless_extra_flags)
            wavel_key_flag = "--wavelength-key=WAVEL " if merge_mode == "poly" else ""
            chained_merge = (
                f"set -e; "
                f"MKEYS=$(python {shlex.quote(prep_script)}) && "
                f'careless {merge_mode} --jit-compile --iterations={careless_iters} --merge-half-datasets {wavel_key_flag}--studentt-likelihood-dof={studentt_dof} {extra_flags_str} "$MKEYS" {shlex.quote(unmerged_mtz)} {shlex.quote(out_prefix)} && '
                f"python {shlex.quote(post_script)}"
            )
            inner_cmd = ["bash", "-c", chained_merge]
        else:
            inner_cmd = [
                "python",
                "-u",
                "-m",
                "subhkl.io.parser",
                "indexer-visualize",
                solve_path,
                "--instrument",
                str(inst),
                "--output-dir",
                tempfile.gettempdir(),
                "--dpi",
                "72",
            ]

        container_image = (
            "ghcr.io/jglaser/careless:cuda"
            if stage_id == "merge"
            else get_container_image()
        )
        image_fallback_note: str | None = None
        podman_bin = shutil.which("podman")
        if podman_bin:
            if stage_id != "merge":
                container_image, image_fallback_note = self._resolve_local_podman_image(
                    container_image,
                    podman_bin=podman_bin,
                )
            raw_dirs = {
                d
                for p in raw_nxs_local
                for d in (
                    os.path.dirname(os.path.abspath(p)),
                    os.path.dirname(os.path.realpath(p)),
                )
            }
            mount_dirs = sorted(
                {
                    os.path.dirname(merged_path),
                    os.path.dirname(solve_path),
                    work_dir,
                    staging_root,
                    tempfile.gettempdir(),
                    *raw_dirs,
                }
            )
            cmd = [
                "podman",
                "run",
                "--rm",
                "--pids-limit=-1",
                "--security-opt",
                "label=disable",
                "--device",
                "nvidia.com/gpu=all",
                "-e",
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}",
                "-e",
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "JAX_PLATFORMS=",
                "-e",
                "MKL_NUM_THREADS=1",
            ]
            for d in mount_dirs:
                if os.path.isdir(d):
                    cmd.extend(["-v", f"{d}:{d}"])
            local_subhkl_pkg = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            if os.path.isdir(local_subhkl_pkg) and stage_id != "merge":
                cmd.extend(
                    [
                        "-v",
                        f"{local_subhkl_pkg}:/usr/local/lib/python3.13/site-packages/subhkl:ro",
                    ]
                )
            cmd.append(container_image)
            cmd.extend(inner_cmd)
        else:
            if inner_cmd[0] == "bash":
                cmd = inner_cmd
            else:
                cmd = [sys.executable] + inner_cmd[1:]

        gpu_env = os.environ.copy()
        gpu_env["JAX_PLATFORMS"] = ""
        gpu_env["TMPDIR"] = staging_root
        gpu_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        gpu_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        gpu_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        gpu_env.setdefault("OMP_NUM_THREADS", "1")
        gpu_env.setdefault("MKL_NUM_THREADS", "1")
        try:
            gpu_info = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=2.0,
                )
                .strip()
                .splitlines()[0]
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            gpu_info = "CPU fallback"
            gpu_env["JAX_PLATFORMS"] = "cpu"

        seq = int(getattr(self, "_local_job_seq", 0)) + 1
        self._local_job_seq = seq
        job_id = f"gpu-{int(time.time() * 1000)}-{seq}"
        container_name: str | None = None
        if cmd and cmd[0] == "podman":
            container_name = f"subhkl-{job_id}"
            cmd.insert(3, f"--name={container_name}")

        from subhkl.io.parser import detect_runner_hostname

        host_tag = detect_runner_hostname()
        cmd_str = " ".join(cmd)
        header_lines: list[str] = []
        if fallback_reason and existing_job_record is None:
            header_lines.append(f"[ndip] Note: {fallback_reason}")
        if image_fallback_note:
            header_lines.append(f"[container] Note: {image_fallback_note}")
        header_lines.append(f"[pulsar] Runner host: {host_tag}")
        header_lines.append(
            f"[container] Host={host_tag} | GPU={gpu_info} | Image={container_image}"
        )
        if stage_id == "merge":
            header_lines.append(
                f"$ careless {merge_mode} --jit-compile --iterations={careless_iters} "
                f"--merge-half-datasets {wavel_key_flag}--studentt-likelihood-dof={studentt_dof} "
                f"{extra_flags_str} <METADATA_KEYS> input_unmerged.mtz careless\n"
            )
        else:
            header_lines.append(f"$ {cmd_str}\n")

        if existing_job_record is not None:
            job_record = existing_job_record
            job_record["job_id"] = job_id
            job_record["container_name"] = container_name
            job_record["mode"] = "container_gpu"
            job_record["state"] = "running"
            job_record["runner_host"] = host_tag
            job_record["command_line"] = cmd_str
            job_record.setdefault("params", dict(params))
            job_record.setdefault("instrument", str(inst))
            if raw_nxs_local and not job_record.get("raw_nxs"):
                job_record["raw_nxs"] = list(raw_nxs_local)
            job_record["stdout"] = (
                job_record.get("stdout", "") + "\n".join(header_lines)
            )[-24000:]
        else:
            job_record = {
                "job_id": job_id,
                "client_job_id": job_id,
                "container_name": container_name,
                "tool_id": f"subhkl::{stage_id}",
                "stage_id": stage_id,
                "mode": "container_gpu",
                "state": "running",
                "runner_host": host_tag,
                "command_line": cmd_str,
                "stdout": "\n".join(header_lines),
                "stderr": "",
                "ladder_rung_idx": 0 if stage_id == "solve" else 4,
                "outputs": [
                    {"id": job_id, "name": os.path.basename(out_h5), "extension": "h5"}
                ],
                "updated_payload_ready": False,
                "hydrate_only": bool(preserve_downstream),
                "n_runs": len(raw_nxs_local) if stage_id == "reduce" else 0,
                "params": dict(params),
                "raw_nxs": list(raw_nxs_local),
                "instrument": str(inst),
                "work_dir": work_dir,
                "started_at": time.time(),
            }
            self.active_job = job_record
        self._sync_job_timing(job_record)

        if container_name:
            self._active_container_name = container_name
        if not hasattr(self, "_active_procs"):
            self._active_procs = {}

        if stage_id == "solve":
            raw_bin_rec = params.get("binning")
            solve_bin_rec = (
                int(raw_bin_rec)
                if raw_bin_rec not in (None, "", "auto", "default")
                and int(raw_bin_rec) >= 1
                else None
            )
            job_record["images_h5_path"] = merged_path
            job_record["solve_params"] = {
                "instrument": str(inst),
                "space_group": str(sg),
                "cell": [float(x) for x in raw_cell[:6]],
                "wavelength_range": (float(wl[0]), float(wl[1])),
                "d_min": float(params.get("d_min") or round(0.5 * float(wl[0]), 2)),
                "binning": solve_bin_rec,
            }

        def _stream_proc() -> None:
            proc: subprocess.Popen[bytes] | None = None
            try:
                if job_record.get("cancel_requested"):
                    job_record["state"] = "deleted"
                    return
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=gpu_env,
                    start_new_session=True,
                )
                self._active_procs[job_id] = proc
                self._active_proc = proc
                if job_record.get("cancel_requested"):
                    self._terminate_local_container_or_proc(container_name, proc)
                    job_record["state"] = "deleted"
                    return
                assert proc.stdout is not None
                blind_src = out_h5[:-3] + ".blind.h5"
                stages_src = out_h5[:-3] + ".stages.h5"
                blind_dst = solve_dest[:-3] + ".blind.h5"
                last_live_key: tuple[str, int] = ("", 0)
                partial_seq = 0
                initial_prefix = job_record.get("stdout", "")
                if initial_prefix and not initial_prefix.endswith("\n"):
                    initial_prefix += "\n"
                raw_stream = ""
                fd = proc.stdout.fileno()
                while True:
                    chunk_bytes = os.read(fd, 4096)
                    if not chunk_bytes:
                        break
                    chunk_str = chunk_bytes.decode("utf-8", errors="replace")
                    raw_stream += chunk_str
                    if len(raw_stream) > 96000 and "\n" in raw_stream[:-8192]:
                        cut_nl = raw_stream.rfind("\n", 0, len(raw_stream) - 8192)
                        if cut_nl > 0:
                            raw_stream = (
                                self._format_terminal_stream(raw_stream[:cut_nl])
                                + "\n"
                                + raw_stream[cut_nl + 1 :]
                            )
                    formatted_stream = self._format_terminal_stream(raw_stream)
                    job_record["stdout"] = (initial_prefix + formatted_stream)[-24000:]
                    low = chunk_str.lower()
                    if stage_id == "solve":
                        if (
                            "blind band search" in low
                            or "band table" in low
                            or "exhaustive rung" in low
                            or "rung 13 zones" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 1
                            )
                        if "rung 30 zones" in low or "rung 60 zones" in low:
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 2
                            )
                        if (
                            "rung 145 zones" in low
                            or "joint band refinement" in low
                            or "candidate " in low
                            or "full-data scores" in low
                            or "ewald pre-polish" in low
                            or "global" in low
                            or "60-zone" in low
                            or "[stage 0" in low
                            or "120" in low
                            or "cell" in low
                            or "[stage 1" in low
                            or "[stage 2" in low
                            or "[stage 3" in low
                            or "[stage 4" in low
                        ):
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 3
                            )
                        if "band refinement: converged" in low:
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 4
                            )
                        active_live_src = None
                        if (
                            os.path.isfile(stages_src)
                            and os.path.getsize(stages_src) > 0
                        ):
                            active_live_src = stages_src
                            job_record["ladder_rung_idx"] = max(
                                job_record["ladder_rung_idx"], 3
                            )
                        elif (
                            os.path.isfile(blind_src) and os.path.getsize(blind_src) > 0
                        ):
                            active_live_src = blind_src
                        if active_live_src is not None:
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                cur_mtime_ns = int(os.stat(active_live_src).st_mtime_ns)
                                cur_key = (active_live_src, cur_mtime_ns)
                                if cur_key != last_live_key:
                                    last_live_key = cur_key
                                    shutil.copy2(active_live_src, blind_dst)
                                    job_record["partial_h5_path"] = blind_dst
                                    from subhkl.viz.rainbow_zones import (
                                        build_rainbow_zone_payload,
                                    )

                                    ov_frame = int(self.current_frame_index or 0)
                                    p_blind = build_rainbow_zone_payload(
                                        replay_h5=blind_dst,
                                        images_h5=merged_path,
                                        instrument=str(inst),
                                        space_group=str(sg),
                                        cell=None,
                                        wavelength_range=(
                                            float(wl[0]),
                                            float(wl[1]),
                                        ),
                                        max_zone_index=3,
                                        max_zones=60,
                                        include_panels=True,
                                        include_panel_images=False,
                                        include_peaks=True,
                                        n_arc=720,
                                        frame_index=ov_frame,
                                    )
                                    if p_blind.get("zones"):
                                        partial_seq += 1
                                        job_record["partial_overlay"] = {
                                            "seq": partial_seq,
                                            "frame_index": int(
                                                p_blind.get("frame_index", ov_frame)
                                            ),
                                            "zones": p_blind.get("zones", []),
                                            "peaks": p_blind.get("peaks", []),
                                            "sample_peaks": p_blind.get(
                                                "sample_peaks", []
                                            ),
                                            "panels": p_blind.get("panels", []),
                                            "cell": p_blind.get("cell"),
                                            "ub_matrix": p_blind.get("ub_matrix"),
                                            "goniometer_R": p_blind.get("goniometer_R"),
                                            "ladder_steps": p_blind.get(
                                                "ladder_steps", []
                                            ),
                                        }
                rc = proc.wait()
                final_h5: str | None = None
                if rc == 0 and not job_record.get("cancel_requested"):
                    job_record["ladder_rung_idx"] = 0 if stage_id == "reduce" else 4
                    final_h5 = out_h5
                    if stage_id == "reduce" and os.path.isfile(out_h5):
                        merged_dest = os.path.join(work_dir, merged_label)
                        merged_link = os.path.join(work_dir, "merged.h5")
                        try:
                            self._stamp_merged_metadata(out_h5, job_record)
                            shutil.move(out_h5, merged_dest)
                            if os.path.abspath(merged_dest) != os.path.abspath(
                                merged_link
                            ):
                                try:
                                    if os.path.lexists(merged_link):
                                        os.remove(merged_link)
                                    os.link(merged_dest, merged_link)
                                except OSError:
                                    shutil.copy2(merged_dest, merged_link)
                            final_h5 = merged_dest
                            _chmod_world_rw(merged_dest, is_dir=False)
                            _chmod_world_rw(merged_link, is_dir=False)
                            job_record["stdout"] += (
                                f"\n[subhkl] Saved {merged_label} -> {merged_dest}\n"
                            )
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {merged_dest}: {copy_exc}\n"
                            )

                        # Keep multi-hundred-MB / multi-GB reduced scan HDF5 in local NVMe staging;
                        # never upload merged.h5 to NDIP / Rucio object storage.
                        if self.is_connected and os.path.isfile(final_h5):
                            sz_mb = os.path.getsize(final_h5) / (1024.0 * 1024.0)
                            job_record["stdout"] += (
                                f"[ndip] Kept '{merged_label}' in local staging ({sz_mb:.1f} MB; not uploaded to NDIP)\n"
                            )
                    elif stage_id == "solve" and os.path.isfile(out_h5):
                        try:
                            shutil.copy2(out_h5, solve_dest)
                            self._stamp_derived_h5_provenance(
                                solve_dest, merged_path=merged_path, params=params
                            )
                            _chmod_world_rw(solve_dest, is_dir=False)
                            for suffix in (".stages.h5", ".blind.h5"):
                                src_sub = out_h5[:-3] + suffix
                                dst_sub = solve_dest[:-3] + suffix
                                if os.path.isfile(src_sub):
                                    shutil.copy2(src_sub, dst_sub)
                                    _chmod_world_rw(dst_sub, is_dir=False)
                            final_h5 = solve_dest
                            job_record["stdout"] += (
                                f"\n[subhkl] Saved solve.h5 -> {solve_dest}\n"
                            )
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving {solve_dest}: {copy_exc}\n"
                            )

                        if self.is_connected and os.path.isfile(final_h5):

                            def _bg_upload_solve(_f_h5: str = final_h5) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    up_id = self._upload_compact_ndip_artifact(
                                        store,
                                        _f_h5,
                                        "h5",
                                        job_record=bg_rec,
                                    )
                                    if up_id:
                                        job_record["outputs"] = [
                                            {
                                                "id": up_id,
                                                "name": "solve.h5",
                                                "extension": "h5",
                                            }
                                        ]

                            threading.Thread(
                                target=_bg_upload_solve, daemon=True
                            ).start()
                    elif stage_id == "integrate" and os.path.isfile(out_h5):
                        try:
                            if os.path.isfile(pred_tmp_h5):
                                shutil.copy2(pred_tmp_h5, pred_dest)
                                self._stamp_derived_h5_provenance(
                                    pred_dest, merged_path=merged_path, params=params
                                )
                                _chmod_world_rw(pred_dest, is_dir=False)
                                job_record["stdout"] += (
                                    f"\n[subhkl] Saved predictor.h5 -> {pred_dest}\n"
                                )
                            shutil.copy2(out_h5, integrated_dest)
                            self._stamp_derived_h5_provenance(
                                integrated_dest, merged_path=merged_path, params=params
                            )
                            _chmod_world_rw(integrated_dest, is_dir=False)
                            final_h5 = integrated_dest
                            job_record["stdout"] += (
                                f"[subhkl] Saved integrated.h5 -> {integrated_dest}\n"
                            )
                            if os.path.isfile(mtz_tmp_path):
                                shutil.copy2(mtz_tmp_path, mtz_dest)
                                _chmod_world_rw(mtz_dest, is_dir=False)
                                job_record["stdout"] += (
                                    f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                                )
                        except _NOVA_RECOVERABLE_ERRORS as copy_exc:
                            job_record["stdout"] += (
                                f"\n[subhkl] Warning saving integrated outputs: {copy_exc}\n"
                            )

                        # Persist compact derived integration artifacts (`integrated.h5`,
                        # `predictor.h5`, and `reflections.mtz` — ~hundreds of KB each) to
                        # NDIP History so future app instances can reload/restart Stage 3/4.
                        if self.is_connected and (
                            os.path.isfile(integrated_dest) or os.path.isfile(mtz_dest)
                        ):

                            def _bg_upload_integrate(
                                _i_h5: str = integrated_dest,
                                _p_h5: str = pred_dest,
                                _m_mtz: str = mtz_dest,
                            ) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    outs_int: list[dict[str, Any]] = []
                                    for _fp_u, _fn_u, _ext_u in (
                                        (_i_h5, "integrated.h5", "h5"),
                                        (_p_h5, "predictor.h5", "h5"),
                                        (_m_mtz, "reflections.mtz", "mtz"),
                                    ):
                                        if os.path.isfile(_fp_u):
                                            _uid = self._upload_compact_ndip_artifact(
                                                store,
                                                _fp_u,
                                                _ext_u,
                                                job_record=bg_rec,
                                            )
                                            if _uid:
                                                outs_int.append(
                                                    {
                                                        "id": _uid,
                                                        "name": _fn_u,
                                                        "extension": _ext_u,
                                                    }
                                                )
                                    if outs_int:
                                        job_record["outputs"] = outs_int

                            threading.Thread(
                                target=_bg_upload_integrate, daemon=True
                            ).start()
                    elif stage_id == "merge":
                        target_int = (
                            integrated_dest
                            if os.path.isfile(integrated_dest)
                            else self.replay_h5
                        )
                        try:
                            if os.path.isfile(target_int):
                                self._stamp_careless_merge_into_h5(
                                    target_int, work_dir, job_record
                                )
                                final_h5 = target_int
                        except _NOVA_RECOVERABLE_ERRORS as merge_exc:
                            job_record["stdout"] += (
                                f"\n[careless] Warning stamping merge results into HDF5: {merge_exc}\n"
                            )
                        for _mfn in (
                            "merged.mtz",
                            "predictions.mtz",
                            "xval.mtz",
                            "careless_metrics.json",
                            "merge_table.csv",
                            "merge_table.html",
                        ):
                            _chmod_world_rw(os.path.join(work_dir, _mfn), is_dir=False)
                        if self.is_connected:

                            def _bg_upload_merge(_wdir: str = work_dir) -> None:
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    store = self.get_datastore("IMAGINE-X SubHKL")
                                    bg_rec: dict[str, Any] = {"stdout": ""}
                                    outs_up = []
                                    for fn, ext in (
                                        ("integrated.h5", "h5"),
                                        ("merged.mtz", "mtz"),
                                        ("xval.mtz", "mtz"),
                                        ("careless_metrics.json", "json"),
                                        ("merge_table.csv", "csv"),
                                        ("merge_table.html", "html"),
                                    ):
                                        fp = os.path.join(_wdir, fn)
                                        if os.path.isfile(fp):
                                            did = self._upload_compact_ndip_artifact(
                                                store,
                                                fp,
                                                ext,
                                                job_record=bg_rec,
                                            )
                                            if did:
                                                outs_up.append(
                                                    {
                                                        "id": did,
                                                        "name": fn,
                                                        "extension": ext,
                                                    }
                                                )
                                    if outs_up:
                                        job_record["outputs"] = outs_up

                            threading.Thread(
                                target=_bg_upload_merge, daemon=True
                            ).start()

                    if os.path.isfile(final_h5):
                        job_record["output_h5_path"] = final_h5
                        if callable(on_complete_h5):
                            on_complete_h5(final_h5)
                            job_record["updated_payload_ready"] = True
                    job_record["stdout"] += (
                        f"[subhkl] Stage '{stage_id}' completed (exit 0).\n"
                    )
                    job_record["state"] = "ok"
                elif job_record.get("cancel_requested"):
                    job_record["state"] = "deleted"
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += (
                        f"\n[subhkl] Process exited with code {rc}\n"
                    )
            except _NOVA_RECOVERABLE_ERRORS as exc:
                if job_record.get("cancel_requested"):
                    job_record["state"] = "deleted"
                else:
                    job_record["state"] = "error"
                    job_record["stderr"] += f"\n[subhkl] Error: {exc}\n"
            finally:
                self._sync_job_timing(job_record)
                if job_record.get("cancel_requested") and container_name:
                    self._terminate_local_container_or_proc(container_name, proc)
                # Remove transient per-job staging artifacts in TMPDIR once copied to work_dir
                # (keeping final_h5 until process exit only if copy fallback left final_h5 == out_h5).
                keep_abs = (
                    os.path.abspath(final_h5)
                    if isinstance(locals().get("final_h5"), str) and final_h5
                    else ""
                )
                for _tmp_item in job_stage_temps:
                    if keep_abs and os.path.abspath(_tmp_item) == keep_abs:
                        continue
                    _remove_temp_artifact_now(_tmp_item)
                if hasattr(self, "_active_procs"):
                    self._active_procs.pop(job_id, None)
                if getattr(self, "_active_proc", None) is proc:
                    self._active_proc = None
                if getattr(self, "_active_container_name", None) == container_name:
                    self._active_container_name = None

        threading.Thread(target=_stream_proc, daemon=True).start()
        return {
            "stage_id": stage_id,
            "mode": "local_stream",
            "galaxy_url": self.galaxy_url,
            "history_id": self.history_id,
            "job": job_record,
            "status": "submitted",
        }

    def _poll_galaxy_job(
        self,
        job_record: dict[str, Any],
        on_complete_h5: Any = None,
        nova_job: Any = None,
        *,
        final_step: bool = True,
        work_dir: str | None = None,
    ) -> None:
        job_id = job_record.get("job_id")
        if not job_id:
            return
        step_counter = 0
        poll_errors = 0
        new_state_ticks = 0
        last_live_token = ""
        partial_seq = 0
        base_log = job_record.get("stdout", "")
        is_localhost_galaxy = bool(
            self.galaxy_url
            and any(h in self.galaxy_url for h in ("127.0.0.1", "localhost"))
        )

        while True:
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                break
            time.sleep(1.5)
            if job_record.get("cancel_requested"):
                job_record["state"] = "deleted"
                break
            try:
                # Fetch job status & live Pulsar console_output stream from NDIP Galaxy API
                jinfo = self._api_request("GET", f"/api/jobs/{job_id}?full=true")
                poll_errors = 0
                state = jinfo.get("state", "running")
                if state == "new":
                    new_state_ticks += 1
                    if is_localhost_galaxy and new_state_ticks >= 10:
                        self._pulsar_queue_stalled_until = time.time() + 300.0
                        with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                            self._api_request("DELETE", f"/api/jobs/{job_id}")
                        raise TimeoutError(
                            f"Galaxy Pulsar queue did not pick up job {job_id} after 15s (stuck in state=new)"
                        )
                else:
                    new_state_ticks = 0
                if state != "ok":
                    job_record["state"] = state
                else:
                    job_record["state"] = "running"
                if jinfo.get("command_line"):
                    job_record["command_line"] = jinfo["command_line"]

                console_out = ""
                console_err = ""
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    cinfo = self._api_request(
                        "GET",
                        f"/api/jobs/{job_id}/console_output?stdout_position=0&stdout_length=65536&stderr_position=0&stderr_length=65536",
                    )
                    if isinstance(cinfo, dict):
                        console_out = cinfo.get("stdout") or ""
                        console_err = cinfo.get("stderr") or ""

                raw_stdout_src = (
                    console_out or jinfo.get("tool_stdout") or jinfo.get("stdout") or ""
                )
                out_txt = self._format_terminal_stream(raw_stdout_src)
                err_txt = self._format_terminal_stream(
                    console_err or jinfo.get("tool_stderr") or jinfo.get("stderr") or ""
                )
                import re as _re_h

                r_host = ""
                m_host = _re_h.search(
                    r"\[(?:pulsar|subhkl|careless)\]\s*Runner host:\s*([A-Za-z0-9._-]+)",
                    f"{raw_stdout_src}\n{console_err}",
                    _re_h.IGNORECASE,
                )
                if m_host:
                    r_host = m_host.group(1).strip()
                if not r_host:
                    for jm in jinfo.get("job_metrics") or []:
                        if not isinstance(jm, dict):
                            continue
                        jm_name = str(jm.get("name") or "").strip().lower()
                        jm_raw = str(
                            jm.get("raw_value") or jm.get("value") or ""
                        ).strip()
                        if not jm_raw:
                            continue
                        if jm_name in (
                            "pulsar_hostname",
                            "host_hostname",
                            "slurm_nodelist",
                        ):
                            r_host = jm_raw.split(".")[0].strip()
                            break
                        if jm_name == "uname":
                            parts = jm_raw.split()
                            cand = parts[1].split(".")[0] if len(parts) >= 2 else ""
                            if cand and not _re_h.fullmatch(r"[0-9a-f]{12}", cand):
                                r_host = cand
                                break
                if r_host:
                    job_record["runner_host"] = r_host
                if job_record.get("stage_id") == "merge":
                    host_line = (
                        f"[pulsar] Runner host: {job_record['runner_host']}\n"
                        if job_record.get("runner_host")
                        and "runner host:" not in out_txt.lower()
                        else ""
                    )
                    job_record["stdout"] = f"{base_log}{host_line}{out_txt}"[-24000:]
                else:
                    raw_cmd = str(job_record.get("command_line") or "").strip()
                    cmd_header = (
                        f"$ {raw_cmd}\n"
                        if raw_cmd and "\n" not in raw_cmd and "PYEOF" not in raw_cmd
                        else ""
                    )
                    job_record["stdout"] = f"{base_log}{cmd_header}{out_txt}"[-24000:]
                if err_txt:
                    job_record["stderr"] = err_txt[-16000:]

                if job_record.get("stage_id") == "solve" and state in (
                    "queued",
                    "running",
                ):
                    low_out = out_txt.lower()
                    cur_rung = int(job_record.get("ladder_rung_idx") or 0)
                    if (
                        "blind band search" in low_out
                        or "band table" in low_out
                        or "exhaustive rung" in low_out
                        or "rung 13 zones" in low_out
                    ):
                        cur_rung = max(cur_rung, 1)
                    if "rung 30 zones" in low_out or "rung 60 zones" in low_out:
                        cur_rung = max(cur_rung, 2)
                    if (
                        "rung 145 zones" in low_out
                        or "joint band refinement" in low_out
                        or "candidate " in low_out
                        or "full-data scores" in low_out
                        or "[stage " in low_out
                    ):
                        cur_rung = max(cur_rung, 3)
                    if "band refinement: converged" in low_out:
                        cur_rung = max(cur_rung, 4)
                    if cur_rung == 0 and state == "running":
                        step_counter = min(3, step_counter + 1)
                        cur_rung = step_counter
                    job_record["ladder_rung_idx"] = cur_rung

                    if "[subhkl-live-state]" in raw_stdout_src:
                        live_tokens = [
                            ln.strip().split("[subhkl-live-state]", 1)[1].strip()
                            for ln in raw_stdout_src.splitlines()
                            if "[subhkl-live-state]" in ln
                        ]
                        if live_tokens and live_tokens[-1] != last_live_token:
                            last_live_token = live_tokens[-1]
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                blind_dst = os.path.join(
                                    work_dir or tempfile.gettempdir(),
                                    "solve.blind.h5",
                                )
                                register_temp_artifact(blind_dst)
                                if self._materialize_live_state_h5(
                                    last_live_token, blind_dst
                                ):
                                    job_record["partial_h5_path"] = blind_dst
                                    partial_seq += 1
                                    self.apply_active_solve_overlay(
                                        {},
                                        frame_index=int(self.current_frame_index or 0),
                                    )
                                    if isinstance(
                                        job_record.get("partial_overlay"), dict
                                    ):
                                        job_record["partial_overlay"]["seq"] = (
                                            partial_seq
                                        )
                if state in ("ok", "error", "deleted", "paused"):
                    if state == "ok":
                        try:
                            st_id_cur = str(job_record.get("stage_id") or "")
                            job_record["outputs"] = self._reconcile_galaxy_job_outputs(
                                job_record.get("outputs") or [],
                                jinfo.get("outputs"),
                                stage_id=st_id_cur,
                            )
                            job_record["ladder_rung_idx"] = (
                                0 if st_id_cur == "reduce" else 4
                            )
                            if not final_step:
                                job_record["stdout"] += (
                                    f"\n[galaxy -> pulsar] Step job {job_id} finished.\n"
                                )
                            h5_outs = (
                                []
                                if st_id_cur == "merge"
                                else [
                                    o
                                    for o in job_record.get("outputs", [])
                                    if o.get("extension")
                                    in (
                                        "h5",
                                        "hdf5",
                                        "json",
                                        "data",
                                        "auto",
                                        "binary",
                                        None,
                                    )
                                    and not str(o.get("name", ""))
                                    .lower()
                                    .endswith(
                                        (
                                            ".mtz",
                                            ".png",
                                            ".pdf",
                                            ".log",
                                            ".html",
                                            ".csv",
                                        )
                                    )
                                    and "mtz" not in str(o.get("name", "")).lower()
                                    and "careless_metrics"
                                    not in str(o.get("name", "")).lower()
                                    and "metrics_json"
                                    not in str(o.get("name", "")).lower()
                                    and "merge_table"
                                    not in str(o.get("name", "")).lower()
                                ]
                            )
                            mtz_outs = [
                                o
                                for o in job_record.get("outputs", [])
                                if o.get("extension") == "mtz"
                                or "mtz" in str(o.get("name", "")).lower()
                            ]
                            if h5_outs:
                                ds_id = h5_outs[0]["id"]
                                st_id = str(job_record.get("stage_id") or "")
                                default_lbl = (
                                    "merged.h5"
                                    if st_id == "reduce"
                                    else (
                                        "solve.h5"
                                        if st_id == "solve"
                                        else (
                                            "predictor.h5"
                                            if (st_id == "integrate" and not final_step)
                                            else "integrated.h5"
                                        )
                                    )
                                )
                                merged_label = str(
                                    job_record.get("merged_label") or default_lbl
                                )
                                out0_ext = str(
                                    h5_outs[0].get("extension") or ""
                                ).lower()
                                out0_name = str(h5_outs[0].get("name") or "").lower()
                                is_json_ptr_out = (
                                    out0_ext == "json" or out0_name.endswith(".json")
                                )
                                hist_rename_lbl = (
                                    re.sub(
                                        r"(_merged)?\.(h5|hdf5)$",
                                        "_scan.ref.json",
                                        merged_label,
                                        flags=re.IGNORECASE,
                                    )
                                    if (is_json_ptr_out and st_id == "reduce")
                                    else merged_label
                                )
                                if (
                                    job_record.get("stage_id") == "reduce"
                                    and final_step
                                    and self.history_id
                                ):
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        self._api_request(
                                            "PUT",
                                            f"/api/histories/{self.history_id}/contents/{ds_id}",
                                            payload={"name": hist_rename_lbl},
                                        )
                                        h5_outs[0]["name"] = hist_rename_lbl
                                        job_record["stdout"] += (
                                            f"[galaxy -> history] Created dataset '{hist_rename_lbl}' (id={ds_id}) in Galaxy history\n"
                                        )
                                target_path = None
                                if (
                                    not final_step
                                    and job_record.get("stage_id") == "reduce"
                                    and job_record.get("reduce_part_path")
                                ):
                                    target_path = str(job_record["reduce_part_path"])
                                elif work_dir and os.path.isdir(work_dir):
                                    if (
                                        not final_step
                                        and job_record.get("stage_id") == "integrate"
                                    ):
                                        target_path = os.path.join(
                                            work_dir, "predictor.h5"
                                        )
                                    elif job_record.get("stage_id") == "integrate":
                                        target_path = os.path.join(
                                            work_dir, "integrated.h5"
                                        )
                                    elif job_record.get("stage_id") == "solve":
                                        target_path = os.path.join(work_dir, "solve.h5")
                                    elif job_record.get("stage_id") == "reduce":
                                        target_path = os.path.join(
                                            work_dir, "merged.h5"
                                        )

                                tmp_path = None
                                cache_gx_path = os.path.join(
                                    tempfile.gettempdir(),
                                    f"subhkl_gx_{ds_id}_{merged_label}",
                                )
                                register_temp_artifact(cache_gx_path)
                                dl_dest = target_path
                                created_temp = False
                                if not dl_dest:
                                    fd, dl_dest = tempfile.mkstemp(
                                        suffix=".h5", prefix="galaxy_solve_"
                                    )
                                    os.close(fd)
                                    register_temp_artifact(dl_dest)
                                    created_temp = True
                                n_synced = 0
                                max_dl_attempts = 3
                                resume_part = (
                                    f"{os.path.abspath(dl_dest)}.part."
                                    f"{os.getpid()}.{threading.get_ident()}"
                                )
                                register_temp_artifact(resume_part)
                                try:
                                    for attempt in range(max_dl_attempts):
                                        try:
                                            n_synced = self._download_dataset_to_file(
                                                ds_id,
                                                dl_dest,
                                                job_record=job_record,
                                                label=merged_label,
                                                timeout=90.0,
                                                resume_part_path=resume_part,
                                            )
                                            if n_synced > 0:
                                                break
                                        except _NOVA_RECOVERABLE_ERRORS as dl_exc:
                                            is_nospc = (
                                                isinstance(dl_exc, OSError)
                                                and getattr(dl_exc, "errno", None)
                                                == errno.ENOSPC
                                            )
                                            if (
                                                is_nospc
                                                or attempt == max_dl_attempts - 1
                                            ):
                                                if created_temp and os.path.lexists(
                                                    dl_dest
                                                ):
                                                    with contextlib.suppress(OSError):
                                                        os.remove(dl_dest)
                                                raise
                                            job_record["stdout"] += (
                                                f"[nova <- galaxy] Sync attempt {attempt + 1}/{max_dl_attempts} "
                                                f"for '{merged_label}' interrupted "
                                                f"({self._clean_err_msg(dl_exc)}); retrying...\n"
                                            )
                                        time.sleep(1.5)
                                finally:
                                    if os.path.lexists(resume_part):
                                        with contextlib.suppress(OSError):
                                            os.remove(resume_part)
                                if n_synced > 0 and os.path.isfile(dl_dest):
                                    from subhkl.io.shared_storage import (
                                        read_shared_pointer,
                                        resolve_input_h5_path,
                                    )

                                    ptr_doc = read_shared_pointer(dl_dest)
                                    if ptr_doc is not None:
                                        resolved_shared_p, _ = resolve_input_h5_path(
                                            dl_dest, must_exist=True
                                        )
                                        abs_sh_p = os.path.abspath(resolved_shared_p)
                                        ptr_cache_p = f"{cache_gx_path}.ref.json"
                                        register_temp_artifact(ptr_cache_p)
                                        with contextlib.suppress(OSError):
                                            shutil.move(dl_dest, ptr_cache_p)
                                        if target_path:
                                            _symlink_shared_into_work_dir(
                                                abs_sh_p, target_path
                                            )
                                            tmp_path = target_path
                                        else:
                                            tmp_path = abs_sh_p
                                        ptr_map = getattr(
                                            self, "_shared_pointer_ds_by_path", {}
                                        )
                                        ptr_map[abs_sh_p] = ds_id
                                        self._shared_pointer_ds_by_path = ptr_map
                                        self._dataset_history_by_id[str(ds_id)] = str(
                                            self.history_id or ""
                                        )
                                        register_shared_intermediate(
                                            abs_sh_p,
                                            network_path=str(
                                                ptr_doc.get("network_path") or ""
                                            ),
                                            dataset_id=str(ds_id),
                                            history_id=str(self.history_id or ""),
                                            work_dir=str(work_dir or ""),
                                            delete_after_use=bool(
                                                ptr_doc.get("delete_after_use")
                                            ),
                                        )
                                        if (
                                            job_record.get("stage_id") == "reduce"
                                            and final_step
                                        ):
                                            self.release_shared_scan_intermediates(
                                                keep_work_dir=work_dir,
                                                keep_paths={abs_sh_p},
                                                job_record=job_record,
                                            )
                                        job_record["stdout"] += (
                                            f"[nova <- shared] Resolved '{ptr_doc.get('network_path')}' -> {abs_sh_p}\n"
                                        )
                                    else:
                                        tmp_path = dl_dest
                                        if target_path:
                                            mb_s = n_synced / (1024.0 * 1024.0)
                                            slim_remote_b = int(
                                                job_record.pop(
                                                    "_last_slim_remote_bytes", 0
                                                )
                                                or 0
                                            )
                                            if slim_remote_b > n_synced:
                                                sz_tag = (
                                                    f", {mb_s:.1f} MB 2D stack synced; "
                                                    f"{slim_remote_b / (1024.0**3):.2f} GB with TOF events in Galaxy"
                                                )
                                            else:
                                                sz_tag = (
                                                    f", {mb_s:.1f} MB"
                                                    if mb_s >= 1.0
                                                    else ""
                                                )
                                            job_record["stdout"] += (
                                                f"[nova <- galaxy] Synced '{merged_label}' (id={ds_id}{sz_tag}) -> {target_path}\n"
                                            )
                                else:
                                    if final_step and target_path:
                                        known_map = getattr(
                                            self, "_dataset_id_by_path", {}
                                        )
                                        known_map[os.path.abspath(target_path)] = ds_id
                                        if (
                                            work_dir
                                            and job_record.get("stage_id") == "reduce"
                                        ):
                                            known_map[
                                                os.path.abspath(
                                                    os.path.join(work_dir, merged_label)
                                                )
                                            ] = ds_id
                                        self._dataset_id_by_path = known_map
                                    if created_temp and os.path.lexists(dl_dest):
                                        with contextlib.suppress(OSError):
                                            os.remove(dl_dest)

                                # Never open shared NFS sideband artifacts (ptr_doc != None / symlinks) in 'r+' mode:
                                # they are write-once immutable after producer fsync+close.
                                if (
                                    ptr_doc is None
                                    and tmp_path
                                    and os.path.isfile(tmp_path)
                                    and not os.path.islink(tmp_path)
                                    and job_record.get("stage_id") == "reduce"
                                    and job_record.get("raw_nxs")
                                    and str(job_record.get("instrument") or "CG4D")
                                    .strip()
                                    .upper()
                                    in ("CG4D", "IMAGINE")
                                ):
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        import h5py
                                        import numpy as np

                                        from subhkl.instrument.goniometer import (
                                            calc_goniometer_rotation_matrix,
                                            get_rotation_data_from_nexus,
                                            get_rotation_data_from_title,
                                        )

                                        raw_nxs_list = job_record["raw_nxs"]
                                        inst_name = str(
                                            job_record.get("instrument") or "CG4D"
                                        )
                                        with h5py.File(tmp_path, "r+") as hf:
                                            if "goniometer/angles" in hf:
                                                cur_ang = np.asarray(
                                                    hf["goniometer/angles"][()]
                                                )
                                                if np.allclose(cur_ang, 0.0):
                                                    new_axes, new_angles, new_names = (
                                                        [],
                                                        [],
                                                        [],
                                                    )
                                                    for nxs_f in raw_nxs_list:
                                                        if nxs_f.startswith(
                                                            tempfile.gettempdir()
                                                        ) and os.path.isfile(nxs_f):
                                                            ax, ang, nms = (
                                                                get_rotation_data_from_nexus(
                                                                    nxs_f, inst_name
                                                                )
                                                            )
                                                        else:
                                                            title_meta = self._lookup_oncat_run_title(
                                                                nxs_f
                                                            )
                                                            ax, ang, nms = (
                                                                get_rotation_data_from_title(
                                                                    title_meta,
                                                                    title_inst
                                                                    := inst_name,
                                                                )
                                                            )
                                                            _ = title_inst
                                                        if ang:
                                                            new_axes.append(ax)
                                                            new_angles.append(ang)
                                                            new_names.append(nms)
                                                    if new_angles and not np.allclose(
                                                        new_angles, 0.0
                                                    ):
                                                        del hf["goniometer/angles"]
                                                        hf["goniometer/angles"] = (
                                                            np.asarray(
                                                                new_angles,
                                                                dtype=np.float64,
                                                            )
                                                        )
                                                        if "goniometer/axes" in hf:
                                                            del hf["goniometer/axes"]
                                                        hf["goniometer/axes"] = (
                                                            np.asarray(
                                                                new_axes,
                                                                dtype=np.float64,
                                                            )
                                                        )
                                                        if "goniometer/names" in hf:
                                                            del hf["goniometer/names"]
                                                        hf["goniometer/names"] = (
                                                            np.asarray(
                                                                new_names, dtype="S"
                                                            )
                                                        )
                                                        new_R = [
                                                            calc_goniometer_rotation_matrix(
                                                                ax_r, ang_r
                                                            )
                                                            for ax_r, ang_r in zip(
                                                                new_axes, new_angles
                                                            )
                                                        ]
                                                        if "goniometer/R" in hf:
                                                            del hf["goniometer/R"]
                                                        hf["goniometer/R"] = np.asarray(
                                                            new_R, dtype=np.float64
                                                        )
                                                        job_record["stdout"] += (
                                                            f"[subhkl] Patched goniometer angles from NeXus/ONCat title into {os.path.basename(tmp_path)}\n"
                                                        )

                                if (
                                    ptr_doc is None
                                    and tmp_path
                                    and os.path.isfile(tmp_path)
                                    and not os.path.islink(tmp_path)
                                    and job_record.get("stage_id") == "reduce"
                                ):
                                    self._stamp_merged_metadata(tmp_path, job_record)

                                # Hard-link into /tmp Galaxy cache once AFTER in-place stamping (0 extra disk writes!)
                                if (
                                    tmp_path
                                    and os.path.isfile(tmp_path)
                                    and not os.path.islink(tmp_path)
                                    and target_path
                                    and not job_record.get("reduce_part_path")
                                ):
                                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                        self._link_or_copy_file(tmp_path, cache_gx_path)

                                if final_step and tmp_path and os.path.isfile(tmp_path):
                                    self.touch_dataset(ds_id)
                                    self.attach_dataset_cell_metadata_from_file(
                                        ds_id,
                                        tmp_path,
                                        fallback_params=job_record.get("params"),
                                    )
                                    known_map = getattr(self, "_dataset_id_by_path", {})
                                    known_map[os.path.abspath(tmp_path)] = ds_id
                                    if os.path.isfile(cache_gx_path):
                                        known_map[os.path.abspath(cache_gx_path)] = (
                                            ds_id
                                        )
                                    if (
                                        work_dir
                                        and job_record.get("stage_id") == "reduce"
                                    ):
                                        lbl_path = os.path.join(work_dir, merged_label)
                                        known_map[os.path.abspath(lbl_path)] = ds_id
                                        with contextlib.suppress(
                                            *_NOVA_RECOVERABLE_ERRORS
                                        ):
                                            if os.path.islink(tmp_path):
                                                _symlink_shared_into_work_dir(
                                                    os.path.realpath(tmp_path), lbl_path
                                                )
                                            else:
                                                self._link_or_copy_file(
                                                    tmp_path, lbl_path
                                                )
                                    self._dataset_id_by_path = known_map
                                    job_record["output_h5_path"] = tmp_path
                                    if callable(on_complete_h5):
                                        on_complete_h5(tmp_path)
                                        job_record["updated_payload_ready"] = True
                            if final_step and job_record.get("stage_id") == "merge":
                                pf_cand = str(
                                    (job_record.get("params") or {}).get("peaks_file")
                                    or ""
                                )
                                if not work_dir:
                                    cand_dir = (
                                        os.path.dirname(os.path.abspath(pf_cand))
                                        if pf_cand and os.path.isfile(pf_cand)
                                        else ""
                                    )
                                    if cand_dir and os.access(cand_dir, os.W_OK):
                                        work_dir = cand_dir
                                    else:
                                        work_dir = tempfile.mkdtemp(
                                            prefix="subhkl_galaxy_merge_"
                                        )
                                        register_temp_artifact(work_dir)
                                with contextlib.suppress(OSError):
                                    os.makedirs(work_dir, exist_ok=True)
                            if (
                                final_step
                                and job_record.get("stage_id") == "merge"
                                and work_dir
                                and os.path.isdir(work_dir)
                            ):
                                for out_d in job_record.get("outputs", []):
                                    oid = out_d.get("id")
                                    if not oid:
                                        continue
                                    dest_name = self._match_careless_merge_output_name(
                                        out_d
                                    )
                                    if dest_name:
                                        try:
                                            try:
                                                bdata = self._api_request(
                                                    "GET",
                                                    f"/api/datasets/{oid}/display?to_ext=data",
                                                    raw=True,
                                                    timeout=90.0,
                                                )
                                            except _NOVA_RECOVERABLE_ERRORS:
                                                bdata = self._api_request(
                                                    "GET",
                                                    f"/api/datasets/{oid}/display",
                                                    raw=True,
                                                    timeout=90.0,
                                                )
                                            dest_p = os.path.join(work_dir, dest_name)
                                            with open(dest_p, "wb") as f:
                                                f.write(bdata)
                                            _chmod_world_rw(dest_p, is_dir=False)
                                            known_map = getattr(
                                                self, "_dataset_id_by_path", {}
                                            )
                                            known_map[os.path.abspath(dest_p)] = oid
                                            known_map[os.path.realpath(dest_p)] = oid
                                            self._dataset_id_by_path = known_map
                                            job_record["stdout"] += (
                                                f"[careless] Saved {dest_name} -> {dest_p}\n"
                                            )
                                        except _NOVA_RECOVERABLE_ERRORS as dl_exc:
                                            job_record["stdout"] += (
                                                f"[careless] Warning downloading {dest_name}: {dl_exc}\n"
                                            )
                                int_target = self._prepare_writable_merge_h5_target(
                                    work_dir, job_record
                                )
                                if int_target and os.path.isfile(int_target):
                                    self._stamp_careless_merge_into_h5(
                                        int_target, work_dir, job_record
                                    )
                                    job_record["output_h5_path"] = int_target
                                    if callable(on_complete_h5):
                                        on_complete_h5(int_target)
                                        job_record["updated_payload_ready"] = True
                                if self.is_connected:

                                    def _bg_upload_merge_tables(
                                        _wdir: str = work_dir,
                                    ) -> None:
                                        with contextlib.suppress(Exception):
                                            store_t = self.get_datastore(
                                                "IMAGINE-X SubHKL"
                                            )
                                            bg_rec_t: dict[str, Any] = {"stdout": ""}
                                            known_ids = getattr(
                                                self, "_dataset_id_by_path", {}
                                            )
                                            for fn_t, ext_t in (
                                                ("merge_table.csv", "csv"),
                                                ("merge_table.html", "html"),
                                            ):
                                                fp_t = os.path.join(_wdir, fn_t)
                                                if (
                                                    os.path.isfile(fp_t)
                                                    and os.path.abspath(fp_t)
                                                    not in known_ids
                                                ):
                                                    _chmod_world_rw(fp_t, is_dir=False)
                                                    self._upload_compact_ndip_artifact(
                                                        store_t,
                                                        fp_t,
                                                        ext_t,
                                                        job_record=bg_rec_t,
                                                    )
                                            int_up_p = os.path.join(
                                                _wdir, "integrated.h5"
                                            )
                                            if os.path.isfile(int_up_p):
                                                _chmod_world_rw(int_up_p, is_dir=False)
                                                self._upload_compact_ndip_artifact(
                                                    store_t,
                                                    int_up_p,
                                                    "h5",
                                                    job_record=bg_rec_t,
                                                )

                                    threading.Thread(
                                        target=_bg_upload_merge_tables, daemon=True
                                    ).start()
                            elif (
                                final_step
                                and mtz_outs
                                and work_dir
                                and os.path.isdir(work_dir)
                            ):
                                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                    mtz_id = mtz_outs[0]["id"]
                                    mtz_bytes = self._api_request(
                                        "GET",
                                        f"/api/datasets/{mtz_id}/display",
                                        raw=True,
                                        timeout=180.0,
                                    )
                                    mtz_dest = os.path.join(work_dir, "reflections.mtz")
                                    with open(mtz_dest, "wb") as f:
                                        f.write(mtz_bytes)
                                    job_record["stdout"] += (
                                        f"[subhkl] Saved reflections.mtz -> {mtz_dest}\n"
                                    )
                        except _NOVA_RECOVERABLE_ERRORS as post_exc:
                            job_record["stdout"] += (
                                f"[nova -> galaxy] Warning fetching job output: {self._clean_err_msg(post_exc)}\n"
                            )
                        finally:
                            if final_step:
                                job_record["stdout"] += (
                                    f"\n[galaxy -> pulsar] Job {job_id} finished (state=ok).\n"
                                )
                                job_record["state"] = "ok"
                    else:
                        misc_info = ""
                        for out_d in job_record.get("outputs", []):
                            oid = out_d.get("id")
                            if not oid:
                                continue
                            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                                dmeta = self._api_request("GET", f"/api/datasets/{oid}")
                                if isinstance(dmeta, dict) and dmeta.get("misc_info"):
                                    misc_info = str(dmeta["misc_info"]).strip()
                                    break
                        if not misc_info and jinfo.get("job_messages"):
                            misc_info = "; ".join(
                                str(m.get("desc") or m)
                                if isinstance(m, dict)
                                else str(m)
                                for m in jinfo["job_messages"]
                            ).strip()
                        if misc_info:
                            if misc_info not in job_record.get("stderr", ""):
                                job_record["stderr"] = (
                                    f"{job_record.get('stderr', '')}{misc_info}\n"
                                )[-16000:]
                            if misc_info not in job_record.get("stdout", ""):
                                job_record["stdout"] += (
                                    f"\n[galaxy -> error] {misc_info}\n"
                                )
                        if state == "error" and not jinfo.get("command_line"):
                            raise RuntimeError(
                                misc_info
                                or f"Galaxy job {job_id} failed before command execution"
                            )
                        job_record["state"] = state
                    break
            except TimeoutError:
                raise
            except _NOVA_RECOVERABLE_ERRORS as exc:
                poll_errors += 1
                if poll_errors < 5:
                    continue
                job_record["error"] = str(exc)
                if job_record.get("state") in ("queued", "running", None):
                    job_record["state"] = "error"
                break

    def run_stage(
        self,
        stage_id: str,
        params_by_tool: dict[str, dict[str, Any]] | None = None,
        *,
        history_name: str = "IMAGINE-X SubHKL",
        wait: bool = True,
    ) -> dict[str, Any]:
        """Run one coarse NDIP pipeline stage in Galaxy for full artifact provenance."""
        if stage_id not in self.STAGE_TOOLS:
            raise ValueError(f"Unknown pipeline stage: {stage_id}")
        tool_ids = self.STAGE_TOOLS[stage_id]
        if not self.is_connected:
            return {
                "stage_id": stage_id,
                "mode": "local_preview",
                "tool_ids": tool_ids,
                "status": "simulated",
            }

        try:
            from nova.galaxy import Parameters, Tool
        except ImportError:
            flat_params: dict[str, Any] = {}
            for pdict in (params_by_tool or {}).values():
                flat_params.update(pdict)
            return self.submit_galaxy_stage(stage_id, flat_params)

        datastore = self.get_datastore(history_name)
        outputs_summary: list[dict[str, Any]] = []
        for tool_id in tool_ids:
            tool = Tool(tool_id)
            params = Parameters()
            for k, v in (params_by_tool or {}).get(tool_id, {}).items():
                params.add_input(k, v)
            outputs = tool.run(datastore, params, wait=wait)
            outputs_summary.append(
                {
                    "tool_id": tool_id,
                    "job_id": getattr(getattr(tool, "_job", None), "id", None),
                    "outputs": [getattr(o, "id", str(o)) for o in (outputs or [])],
                }
            )
        return {
            "stage_id": stage_id,
            "mode": "ndip_galaxy",
            "galaxy_url": self.galaxy_url,
            "tool_runs": outputs_summary,
            "status": "completed" if wait else "running",
        }


class RainbowZonesViewModel:
    """MVVM ViewModel managing Rainbow Zones state for NOVA / Trame or HTTP views."""

    def __init__(
        self,
        replay_h5: str | os.PathLike | None = None,
        payload: dict[str, Any] | None = None,
        default_mode: str = "arcade",
        **payload_kwargs: Any,
    ) -> None:
        from subhkl.io.shared_storage import resolve_input_h5_path

        resolved_replay = (
            resolve_input_h5_path(replay_h5, must_exist=False)[0]
            if replay_h5 is not None
            else None
        )
        self.replay_h5 = str(resolved_replay) if resolved_replay is not None else None
        self.default_mode = default_mode
        self.payload_kwargs = dict(payload_kwargs)
        if self.payload_kwargs.get("images_h5"):
            self.payload_kwargs["images_h5"] = resolve_input_h5_path(
                self.payload_kwargs["images_h5"], must_exist=False
            )[0]
        self.pipeline_runner = NovaGalaxyPipelineRunner()
        self.mtz_overlay_path: str | None = None
        self.mtz_overlay_info: dict[str, Any] | None = None
        if payload is not None:
            self.payload = payload
        elif self.replay_h5 is not None:
            self.payload = build_rainbow_zone_payload(
                replay_h5=self.replay_h5, **self.payload_kwargs
            )
        else:
            self.payload = build_rainbow_zone_payload(**self.payload_kwargs)

    @property
    def _payload(self) -> dict[str, Any]:
        return self.payload

    @_payload.setter
    def _payload(self, value: dict[str, Any]) -> None:
        self.payload = value

    def reload(self, **overrides: Any) -> dict[str, Any]:
        from subhkl.viz.rainbow_zones import merge_frame_only_payload

        include_all = bool(overrides.get("include_all_runs_peaks", True))
        prev_payload = (
            self._payload
            if (not include_all and isinstance(self._payload, dict))
            else None
        )
        if "frame_index" in overrides and overrides["frame_index"] is not None:
            self.payload_kwargs["frame_index"] = int(overrides["frame_index"])
        kwargs = {**self.payload_kwargs, **overrides}
        if self.replay_h5 is not None:
            self.payload = build_rainbow_zone_payload(
                replay_h5=self.replay_h5, **kwargs
            )
        else:
            self.payload = build_rainbow_zone_payload(**kwargs)
        if (
            self.mtz_overlay_path
            and os.path.isfile(self.mtz_overlay_path)
            and (
                self.payload.get("u_matrix") is not None
                or self.payload.get("ub_matrix") is not None
            )
        ):
            from subhkl.viz.rainbow_zones import apply_mtz_to_payload

            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                self.payload = apply_mtz_to_payload(
                    self.payload,
                    self.mtz_overlay_path,
                    self.mtz_overlay_info,
                    select_mtz_color=False,
                )
        if prev_payload is not None:
            self.payload = merge_frame_only_payload(prev_payload, self.payload)
        return self.payload

    def to_html(self, mode: str | None = None) -> str:
        return render_rainbow_html(
            self.payload,
            default_mode=mode or self.default_mode,
        )


class RainbowZonesTrameApp:
    """Trame / NOVA application wrapper for Rainbow Zones."""

    def __init__(
        self,
        view_model: RainbowZonesViewModel,
        server: Any = None,
    ) -> None:
        self.vm = view_model
        self.server = server
        self._setup_trame()

    def _setup_trame(self) -> None:
        try:
            from trame.app import get_server
            from trame.ui.html import DivLayout
            from trame.widgets import html as html_widgets
        except ImportError as exc:
            raise RuntimeError(
                "Trame is not installed in this environment; use "
                "serve_rainbow_zones() or export_rainbow_widget() instead."
            ) from exc

        if self.server is None:
            self.server = get_server(client_type="vue3")

        state = self.server.state
        state.trame__title = "SUBHKL // Diffraction Parallax"
        state.trame__favicon = SUBHKL_FAVICON_DATA_URI
        state.rainbow_mode = self.vm.default_mode
        state.rainbow_payload = self.vm.payload
        state.rainbow_iframe_srcdoc = self.vm.to_html()

        with DivLayout(self.server) as layout:
            layout.root.style = "margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;background:#060719;"
            html_widgets.Iframe(
                srcdoc=("rainbow_iframe_srcdoc",),
                style="border:none;width:100%;height:100%;",
            )

    def start(self, host: str = "0.0.0.0", port: int = 8080, **kwargs: Any) -> None:
        self.server.start(
            host=host,
            port=port,
            open_browser=False,
            timeout=0,
            **kwargs,
        )


def serve_rainbow_zones(
    replay_h5: str | os.PathLike | None = None,
    payload: dict[str, Any] | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    default_mode: str = "arcade",
    prefer_trame: bool = False,
    **payload_kwargs: Any,
) -> None:
    """Serve the Rainbow Zones GUI over HTTP (or Trame if requested and installed).

    Compatible with NDIP / Galaxy Interactive Tools (`$EP_PATH`, port 8080) as well
    as local container port-forwarding (`podman run -p 8080:8080 ...`).
    """
    vm = RainbowZonesViewModel(
        replay_h5=replay_h5,
        payload=payload,
        default_mode=default_mode,
        **payload_kwargs,
    )
    if vm.pipeline_runner.is_explicit_galaxy_exec_mode():
        vm.pipeline_runner.require_shared_dir_for_galaxy_mode()

    if prefer_trame:
        try:
            from subhkl.viz.nova_app.views.main_view import MainApp

            nova_app = MainApp(
                peaks_h5=replay_h5,
                images_h5=payload_kwargs.get("images_h5"),
                instrument=payload_kwargs.get("instrument") or "CG4D",
            )
            nova_app.server.start(
                host=host,
                port=port,
                open_browser=False,
                timeout=0,
            )
            return
        except (ImportError, *_NOVA_RECOVERABLE_ERRORS):
            try:
                app = RainbowZonesTrameApp(vm)
                app.start(host=host, port=port)
                return
            except RuntimeError:
                pass

    ep_path = os.environ.get("EP_PATH", "").rstrip("/")
    vm_lock = threading.RLock()
    latest_frame_only_target: list[int | None] = [None]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _serve_payload_locked(self, qs: dict[str, list[str]]) -> None:
            f_idx: int | None = None
            if qs.get("frame"):
                try:
                    f_idx = int(qs["frame"][0])
                    vm.payload_kwargs["frame_index"] = f_idx
                    vm.pipeline_runner.current_frame_index = f_idx
                except (ValueError, TypeError, IndexError):
                    f_idx = None
            preserve_q = qs.get("preserve_params", ["0"])[0] == "1"
            d_min_val: float | None = None
            if qs.get("d_min"):
                with contextlib.suppress(ValueError, TypeError, IndexError):
                    parsed_dmin = float(qs["d_min"][0])
                    if 0.2 <= parsed_dmin <= 20.0:
                        d_min_val = parsed_dmin
                        vm.payload_kwargs["d_min"] = parsed_dmin
            rbin_val: int | None = None
            if qs.get("render_binning"):
                with contextlib.suppress(ValueError, TypeError, IndexError):
                    parsed_rb = int(qs["render_binning"][0])
                    if 1 <= parsed_rb <= 16:
                        rbin_val = parsed_rb
                        vm.payload_kwargs["render_binning"] = parsed_rb
            reloaded = False
            if qs.get("file_path"):
                fp_raw = os.path.abspath(str(qs["file_path"][0]).strip())
                if fp_raw.lower().endswith(".mtz"):
                    if not os.path.isfile(fp_raw):
                        err_body = json.dumps(
                            {
                                "status": "error",
                                "error": f"MTZ file not found: {fp_raw}",
                            }
                        ).encode("utf-8")
                        self.send_response(404)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(err_body)))
                        self.end_headers()
                        self.wfile.write(err_body)
                        return
                    try:
                        from subhkl.viz.rainbow_zones import apply_mtz_to_payload

                        ds_info_mtz = {"name": os.path.basename(fp_raw)}
                        vm.payload = apply_mtz_to_payload(
                            vm.payload or {}, fp_raw, ds_info_mtz
                        )
                        vm.mtz_overlay_path = fp_raw
                        vm.mtz_overlay_info = ds_info_mtz
                        body = json.dumps(vm.payload).encode("utf-8")
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    except _NOVA_RECOVERABLE_ERRORS as exc:
                        err_body = json.dumps(
                            {"status": "error", "error": str(exc)}
                        ).encode("utf-8")
                        self.send_response(400)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(err_body)))
                        self.end_headers()
                        self.wfile.write(err_body)
                        return
                with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                    fp = fp_raw
                    if os.path.isfile(fp):
                        fp_low = os.path.basename(fp).lower()
                        is_reduce_file = (
                            any(k in fp_low for k in ("merged", "scan", "image"))
                            and "careless" not in fp_low
                        )
                        if is_reduce_file:
                            vm.mtz_overlay_path = None
                            vm.mtz_overlay_info = None
                            _, fp = vm.pipeline_runner._resolve_work_dir_and_merged(fp)
                            vm.payload_kwargs["images_h5"] = fp
                            vm.payload_kwargs.pop("d_min", None)
                            vm.replay_h5 = fp
                            vm.reload(images_h5=fp, frame_index=f_idx)
                        else:
                            from subhkl.viz.rainbow_zones import _discover_images_h5

                            if not preserve_q and d_min_val is None:
                                vm.payload_kwargs.pop("d_min", None)
                            work_d_fp, staged_fp = (
                                vm.pipeline_runner.stage_derived_h5_in_work_dir(
                                    fp, name_hint=os.path.basename(fp)
                                )
                            )
                            disc_img = (
                                vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                    staged_fp
                                )
                                or _discover_images_h5(staged_fp)
                            )
                            vm.pipeline_runner._discover_or_fetch_galaxy_predictor_h5(
                                staged_fp
                            )
                            vm.payload_kwargs["images_h5"] = (
                                str(disc_img) if disc_img else None
                            )
                            vm.replay_h5 = staged_fp
                            vm.reload(
                                images_h5=vm.payload_kwargs.get("images_h5"),
                                frame_index=f_idx,
                            )
                            vm.pipeline_runner.invalidate_subsequent_stages(
                                (vm.payload or {}).get("artifact_kind"),
                                work_d_fp,
                            )
                            if (vm._payload or {}).get("intensity_source") != "raw_h5":
                                hyd_job = (
                                    vm.pipeline_runner._maybe_trigger_hydrate_reduce(
                                        vm, vm_lock, staged_fp
                                    )
                                )
                                if hyd_job and isinstance(vm._payload, dict):
                                    vm._payload["hydration_job"] = hyd_job
                        if is_reduce_file:
                            vm.pipeline_runner.invalidate_subsequent_stages(
                                (vm.payload or {}).get("artifact_kind"),
                                os.path.dirname(fp),
                            )
                        reloaded = True
            if qs.get("dataset_id"):
                try:
                    from subhkl.viz.rainbow_zones import (
                        _discover_images_h5,
                        apply_mtz_to_payload,
                        validate_subhkl_ingestible_dataset,
                    )

                    ds_id = str(qs["dataset_id"][0]).strip()
                    fb_dir = (
                        os.path.dirname(os.path.abspath(vm.replay_h5))
                        if vm.replay_h5
                        else None
                    )
                    local_h5, ds_info = vm.pipeline_runner.resolve_dataset_to_local_h5(
                        ds_id, fallback_dir=fb_dir
                    )
                    cur_p = vm.payload or {}
                    has_active_geom = bool(
                        (
                            cur_p.get("artifact_kind") != "empty"
                            and cur_p.get("panels")
                            and (cur_p.get("peaks") or cur_p.get("ub_matrix"))
                        )
                        or qs.get("has_active_geometry", ["0"])[0] == "1"
                    )
                    ingest_kind = validate_subhkl_ingestible_dataset(
                        local_h5,
                        ds_info,
                        has_active_geometry=has_active_geom,
                    )
                    if ingest_kind == "mtz_overlay":
                        vm.payload = apply_mtz_to_payload(cur_p, local_h5, ds_info)
                        vm.mtz_overlay_path = local_h5
                        vm.mtz_overlay_info = ds_info
                        body = json.dumps(vm.payload).encode("utf-8")
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if ingest_kind == "raw_nexus":
                        vm.mtz_overlay_path = None
                        vm.mtz_overlay_info = None
                        raw_body = json.dumps(
                            {
                                "status": "raw_nexus",
                                "dataset_id": ds_id,
                                "name": ds_info.get("name")
                                or os.path.basename(local_h5),
                            }
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(raw_body)))
                        self.end_headers()
                        self.wfile.write(raw_body)
                        return
                    ds_name = str(
                        ds_info.get("name") or os.path.basename(local_h5)
                    ).lower()
                    if (
                        any(k in ds_name for k in ("merged", "scan", "image"))
                        and "careless" not in ds_name
                    ):
                        vm.mtz_overlay_path = None
                        vm.mtz_overlay_info = None
                        work_d_ds, local_h5 = (
                            vm.pipeline_runner._resolve_work_dir_and_merged(local_h5)
                        )
                        vm.payload_kwargs["images_h5"] = local_h5
                        vm.replay_h5 = local_h5
                        vm.reload(images_h5=local_h5, frame_index=f_idx)
                        vm.pipeline_runner.invalidate_subsequent_stages(
                            (vm._payload or {}).get("artifact_kind"),
                            work_d_ds,
                        )
                        vm.pipeline_runner.release_shared_scan_intermediates(
                            keep_work_dir=work_d_ds,
                            keep_paths={os.path.realpath(local_h5)}
                            if os.path.isfile(local_h5)
                            else set(),
                        )
                    else:
                        work_d_ds, staged_ds_h5 = (
                            vm.pipeline_runner.stage_derived_h5_in_work_dir(
                                local_h5, name_hint=ds_name
                            )
                        )
                        disc_img = (
                            vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                staged_ds_h5
                            )
                            or _discover_images_h5(staged_ds_h5)
                        )
                        ds_hid_raw = (ds_info or {}).get("hid")
                        ds_hid = (
                            int(ds_hid_raw)
                            if isinstance(ds_hid_raw, (int, str))
                            and str(ds_hid_raw).isdigit()
                            else None
                        )
                        vm.pipeline_runner._discover_or_fetch_galaxy_predictor_h5(
                            staged_ds_h5, target_hid=ds_hid
                        )
                        vm.payload_kwargs["images_h5"] = (
                            str(disc_img) if disc_img else None
                        )
                        vm.replay_h5 = staged_ds_h5
                        vm.reload(
                            images_h5=(str(disc_img) if disc_img else None),
                            frame_index=f_idx,
                        )
                        vm.pipeline_runner.invalidate_subsequent_stages(
                            (vm._payload or {}).get("artifact_kind"),
                            work_d_ds,
                        )
                        vm.pipeline_runner.release_shared_scan_intermediates(
                            keep_work_dir=work_d_ds,
                            keep_paths={os.path.realpath(str(disc_img))}
                            if disc_img and os.path.isfile(str(disc_img))
                            else set(),
                        )
                        if (vm._payload or {}).get("intensity_source") != "raw_h5":
                            hyd_job = vm.pipeline_runner._maybe_trigger_hydrate_reduce(
                                vm,
                                vm_lock,
                                staged_ds_h5,
                                ds_info=ds_info,
                            )
                            if hyd_job and isinstance(vm._payload, dict):
                                vm._payload["hydration_job"] = hyd_job
                    reloaded = True
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    err_body = json.dumps(
                        {"status": "error", "error": str(exc)}
                    ).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
                    return
            frame_only_q = qs.get("frame_only", ["0"])[0] == "1"
            if not reloaded:
                cur_p = vm._payload
                cur_f = cur_p.get("frame_index") if isinstance(cur_p, dict) else None
                cur_dmin = cur_p.get("d_min") if isinstance(cur_p, dict) else None
                cur_rb = (
                    int(cur_p.get("render_binning", 4))
                    if isinstance(cur_p, dict)
                    else None
                )
                fast_frame_only = bool(
                    frame_only_q
                    and isinstance(cur_p, dict)
                    and (
                        cur_p.get("sample_peaks")
                        or cur_p.get("integrated_sample_peaks")
                        or cur_p.get("predicted_sample_peaks")
                        or len(cur_p.get("frames") or ()) <= 1
                    )
                )
                if (
                    not isinstance(cur_p, dict)
                    or (f_idx is not None and cur_f != f_idx)
                    or (rbin_val is not None and cur_rb != rbin_val)
                    or (
                        d_min_val is not None
                        and (
                            cur_dmin is None or abs(float(cur_dmin) - d_min_val) > 1e-3
                        )
                    )
                ):
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        vm.reload(
                            frame_index=f_idx,
                            include_all_runs_peaks=not fast_frame_only,
                        )
                elif isinstance(cur_p, dict) and d_min_val is not None and preserve_q:
                    cur_p["d_min"] = round(d_min_val, 3)
            else:
                fast_frame_only = False
            with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                vm.pipeline_runner.apply_active_solve_overlay(
                    vm.payload, frame_index=f_idx
                )
            if (
                not reloaded
                and isinstance(vm.payload, dict)
                and "hydration_job" in vm.payload
            ):
                aj_cur = vm.pipeline_runner.active_job
                if (
                    vm.payload.get("intensity_source") == "raw_h5"
                    or not isinstance(aj_cur, dict)
                    or not aj_cur.get("hydrate_only")
                    or aj_cur.get("state") not in ("running", "queued", "new")
                ):
                    vm.payload.pop("hydration_job", None)
                else:
                    vm.payload["hydration_job"] = {
                        **vm.payload["hydration_job"],
                        "job_id": aj_cur.get("job_id"),
                        "client_job_id": aj_cur.get("client_job_id")
                        or aj_cur.get("job_id"),
                        "state": aj_cur.get("state"),
                    }
            if fast_frame_only and isinstance(vm.payload, dict):
                from subhkl.viz.rainbow_zones import build_frame_only_response

                resp_doc = build_frame_only_response(vm.payload)
            else:
                resp_doc = vm.payload
            body = json.dumps(resp_doc, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            req_path = parsed.path
            if ep_path and req_path.startswith(ep_path):
                req_path = req_path[len(ep_path) :] or "/"

            if req_path in ("/favicon.ico", "/favicon.svg", "/logo.png"):
                fav_bytes = SUBHKL_FAVICON_SVG.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Content-Length", str(len(fav_bytes)))
                self.end_headers()
                self.wfile.write(fav_bytes)
                return

            if req_path in ("/api/payload", "/payload.json"):
                if qs.get("textures_only", ["0"])[0] == "1":
                    from subhkl.viz.rainbow_zones import (
                        build_rainbow_panel_textures_only,
                    )

                    f_idx_tex: int | None = None
                    if qs.get("frame"):
                        with contextlib.suppress(ValueError, TypeError, IndexError):
                            f_idx_tex = int(qs["frame"][0])
                    rb_tex = int(vm.payload_kwargs.get("render_binning") or 4)
                    if qs.get("render_binning"):
                        with contextlib.suppress(ValueError, TypeError, IndexError):
                            parsed_rb = int(qs["render_binning"][0])
                            if 1 <= parsed_rb <= 16:
                                rb_tex = parsed_rb
                                vm.payload_kwargs["render_binning"] = parsed_rb
                    tex_doc = build_rainbow_panel_textures_only(
                        replay_h5=vm.replay_h5 or vm.payload_kwargs.get("images_h5"),
                        images_h5=vm.payload_kwargs.get("images_h5"),
                        instrument=vm.payload_kwargs.get("instrument") or "CG4D",
                        render_binning=rb_tex,
                        frame_index=(
                            f_idx_tex
                            if f_idx_tex is not None
                            else int(vm.payload_kwargs.get("frame_index") or 0)
                        ),
                    )
                    if vm_lock.acquire(blocking=False):
                        try:
                            cur_pl = vm._payload
                            if isinstance(cur_pl, dict) and isinstance(
                                cur_pl.get("panels"), list
                            ):
                                cur_pl["render_binning"] = tex_doc["render_binning"]
                        finally:
                            vm_lock.release()
                    body = json.dumps(tex_doc).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                f_only_pre = qs.get("frame_only", ["0"])[0] == "1"
                f_pre: int | None = None
                if qs.get("frame"):
                    with contextlib.suppress(ValueError, TypeError, IndexError):
                        f_pre = int(qs["frame"][0])
                if f_only_pre and f_pre is not None:
                    latest_frame_only_target[0] = f_pre

                vm_lock.acquire()
                try:
                    if (
                        f_only_pre
                        and f_pre is not None
                        and latest_frame_only_target[0] not in (None, f_pre)
                    ):
                        sup_body = json.dumps(
                            {
                                "status": "superseded",
                                "frame_only": True,
                                "frame_index": f_pre,
                            },
                            separators=(",", ":"),
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "application/json; charset=utf-8"
                        )
                        self.send_header("Content-Length", str(len(sup_body)))
                        self.end_headers()
                        self.wfile.write(sup_body)
                        return
                    self._serve_payload_locked(qs)
                finally:
                    vm_lock.release()
                return

            if req_path.startswith("/api/datasets/") and req_path.endswith("/display"):
                ds_id = req_path.split("/")[3]
                try:
                    fb_dir = (
                        os.path.dirname(os.path.abspath(vm.replay_h5))
                        if vm.replay_h5
                        else None
                    )
                    local_h5, _ = vm.pipeline_runner.resolve_dataset_to_local_h5(
                        ds_id, fallback_dir=fb_dir
                    )
                    with open(local_h5, "rb") as fh:
                        h5_bytes = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-hdf5")
                    self.send_header("Content-Length", str(len(h5_bytes)))
                    self.end_headers()
                    self.wfile.write(h5_bytes)
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    err_body = str(exc).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
                return

            if req_path == "/api/artifacts/download":
                ds_id = str(qs.get("dataset_id", [""])[0]).strip()
                fname = str(qs.get("name", [""])[0]).strip()
                try:
                    local_fp, dl_name = (
                        vm.pipeline_runner.resolve_artifact_for_download(
                            dataset_id=ds_id,
                            name=fname,
                            current_h5=vm.replay_h5,
                        )
                    )
                    with open(local_fp, "rb") as fh:
                        raw_bytes = fh.read()
                    safe_dl = dl_name.replace('"', "")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{safe_dl}"',
                    )
                    self.send_header("Content-Length", str(len(raw_bytes)))
                    self.end_headers()
                    self.wfile.write(raw_bytes)
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    err_body = str(exc).encode("utf-8")
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
                return

            if req_path == "/api/pipeline/oncat_catalog":
                cat_doc = vm.pipeline_runner.browse_oncat_catalog(
                    {
                        "facility": str(qs.get("facility", ["HFIR"])[0]).strip(),
                        "instrument": str(qs.get("instrument", ["CG4D"])[0]).strip(),
                        "ipts": str(qs.get("ipts", [""])[0]).strip(),
                    }
                )
                body = json.dumps(cat_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/oncat_auth":
                auth_doc = vm.pipeline_runner.get_oncat_auth_status()
                body = json.dumps(auth_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/status":
                f_idx_st: int | None = None
                if qs.get("frame"):
                    with contextlib.suppress(*_NOVA_RECOVERABLE_ERRORS):
                        f_idx_st = int(qs["frame"][0])
                        vm.payload_kwargs["frame_index"] = f_idx_st
                        vm.pipeline_runner.sync_frame_index(f_idx_st)
                is_poll = qs.get("poll", ["0"])[0] == "1"
                ov_seq_client: int | None = None
                if qs.get("overlay_seq"):
                    with contextlib.suppress(ValueError, TypeError, IndexError):
                        ov_seq_client = int(qs["overlay_seq"][0])
                aj_now = vm.pipeline_runner.active_job
                aj_running = isinstance(aj_now, dict) and aj_now.get("state") in (
                    "queued",
                    "running",
                )
                hist_ds = vm.pipeline_runner.list_history_datasets(
                    use_cached=bool(is_poll and aj_running)
                )
                step_timings_snap = vm.pipeline_runner.get_step_timings_snapshot()
                cur_pl = vm._payload if isinstance(vm._payload, dict) else {}
                aj_resp = vm.pipeline_runner.active_job
                if (
                    is_poll
                    and ov_seq_client is not None
                    and isinstance(aj_resp, dict)
                    and isinstance(aj_resp.get("partial_overlay"), dict)
                ):
                    ov_now = aj_resp["partial_overlay"]
                    ov_seq_now = int(ov_now.get("seq") or 0)
                    ov_frame_now = ov_now.get("frame_index")
                    if ov_seq_now == ov_seq_client and (
                        f_idx_st is None
                        or ov_frame_now is None
                        or int(ov_frame_now) == f_idx_st
                    ):
                        aj_resp = dict(aj_resp)
                        aj_resp["partial_overlay"] = {
                            "seq": ov_seq_now,
                            "frame_index": ov_frame_now,
                            "unchanged": True,
                        }
                status_doc = {
                    "galaxy_connected": vm.pipeline_runner.is_connected,
                    "galaxy_url": vm.pipeline_runner.galaxy_url,
                    "history_id": vm.pipeline_runner.history_id,
                    "history_datasets": hist_ds,
                    "step_artifacts": vm.pipeline_runner.list_step_artifacts(
                        current_h5=vm.replay_h5,
                        history_datasets=hist_ds,
                    ),
                    "active_job": aj_resp,
                    "step_timings": step_timings_snap,
                    "server_time": time.time(),
                    "pipeline_stages": cur_pl.get("pipeline_stages", []),
                    "ladder_steps": cur_pl.get("ladder_steps", []),
                }
                body = json.dumps(status_doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            mode = qs.get("mode", [vm.default_mode])[0]
            body = vm.to_html(mode=mode).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            req_path = parsed.path
            if ep_path and req_path.startswith(ep_path):
                req_path = req_path[len(ep_path) :] or "/"

            if req_path == "/api/pipeline/oncat_auth":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.handle_oncat_auth(req_json)
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc), "message": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/check_neutron_paths":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.check_neutron_paths(req_json)
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc), "message": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/cancel":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    result = vm.pipeline_runner.cancel_active_job(
                        req_json.get("stage_id")
                    )
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/exit":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                should_shutdown = True
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    if not isinstance(req_json, dict):
                        req_json = {}
                    phase_req = str(req_json.get("phase") or "confirm").strip().lower()
                    if phase_req == "prepare" or req_json.get("shutdown") is False:
                        should_shutdown = False
                    cur_imgs_ex = vm.payload_kwargs.get("images_h5") or (
                        vm.payload.get("images_h5")
                        if isinstance(vm.payload, dict)
                        else None
                    )
                    result = vm.pipeline_runner.execute_app_exit(
                        phase=phase_req,
                        current_h5=(str(vm.replay_h5) if vm.replay_h5 else None),
                        images_h5=(str(cur_imgs_ex) if cur_imgs_ex else None),
                        payload=(vm.payload if isinstance(vm.payload, dict) else None),
                        mtz_overlay_info=getattr(vm, "mtz_overlay_info", None),
                    )
                    if not result.get("deferred_cleanup"):
                        with vm_lock:
                            if vm.replay_h5 and not os.path.isfile(str(vm.replay_h5)):
                                vm.replay_h5 = None
                            if cur_imgs_ex and not os.path.isfile(str(cur_imgs_ex)):
                                vm.payload_kwargs["images_h5"] = None
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                with contextlib.suppress(OSError):
                    self.wfile.flush()
                if code == 200 and should_shutdown:
                    srv = self.server

                    def _deferred_httpd_stop() -> None:
                        time.sleep(0.35)
                        with contextlib.suppress(Exception):
                            srv.shutdown()

                    threading.Thread(
                        target=_deferred_httpd_stop,
                        name="subhkl-exit-shutdown",
                        daemon=True,
                    ).start()
                return

            if req_path == "/api/tof_delta_theta":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    from subhkl.viz.rainbow_zones import (
                        compute_live_tof_delta_theta_payload,
                    )

                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    with vm_lock:
                        cur_imgs = vm.payload_kwargs.get("images_h5") or vm.payload.get(
                            "images_h5"
                        )
                        cur_replay = vm.replay_h5
                        def_inst = str(vm.payload.get("instrument") or "IMAGINE")
                        def_sg = str(vm.payload.get("space_group") or "P 1")
                        def_cell = vm.payload.get("cell")
                        def_wl = vm.payload.get("wavelength_range") or [2.0, 4.0]
                        def_dmin = float(vm.payload.get("d_min") or 1.5)
                        def_ki = vm.payload.get("ki") or [0.0, 0.0, 1.0]
                    if not cur_imgs and cur_replay:
                        cur_imgs = (
                            vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                cur_replay
                            )
                        )
                    wl_in = req_json.get("wavelength_range") or def_wl
                    wl_tup = (float(wl_in[0]), float(wl_in[1]))
                    result = compute_live_tof_delta_theta_payload(
                        images_h5=cur_imgs,
                        replay_h5=cur_replay,
                        frame_index=int(req_json.get("frame_index") or 0),
                        d_min=float(req_json.get("d_min") or def_dmin),
                        wavelength_range=wl_tup,
                        ub_matrix=req_json.get("ub_matrix"),
                        R_run=req_json.get("R_run"),
                        sample_offset_lab=req_json.get("sample_offset_lab"),
                        ki=req_json.get("ki") or def_ki,
                        instrument=str(req_json.get("instrument") or def_inst),
                        space_group=str(req_json.get("space_group") or def_sg),
                        cell=req_json.get("cell") or def_cell,
                        detector_configs=req_json.get("detector_configs"),
                        tof_l1=req_json.get("tof_l1"),
                        zone_axes_crystal=req_json.get("zone_axes_crystal"),
                        zone_tolerance_deg=float(
                            req_json.get("zone_tolerance_deg") or 0.75
                        ),
                    )
                    code = 200 if result.get("ok") else 400
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"ok": False, "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/run":
                content_len = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                try:
                    req_json = json.loads(raw_body.decode("utf-8") or "{}")
                    stage_id = req_json.get("stage_id", "solve")
                    params = dict(req_json.get("params") or {})
                    params.setdefault(
                        "instrument", vm.payload.get("instrument", "CG4D")
                    )
                    params.setdefault(
                        "space_group", vm.payload.get("space_group", "P 1")
                    )
                    if all(
                        k in params for k in ("a", "b", "c", "alpha", "beta", "gamma")
                    ):
                        params["cell"] = [
                            float(params["a"]),
                            float(params["b"]),
                            float(params["c"]),
                            float(params["alpha"]),
                            float(params["beta"]),
                            float(params["gamma"]),
                        ]
                    elif vm.payload.get("cell"):
                        params.setdefault("cell", vm.payload.get("cell"))
                    if "wavelength_min" in params and "wavelength_max" in params:
                        params["wavelength_range"] = [
                            float(params["wavelength_min"]),
                            float(params["wavelength_max"]),
                        ]
                    elif vm.payload.get("wavelength_range"):
                        params.setdefault(
                            "wavelength_range", vm.payload.get("wavelength_range")
                        )
                    if vm.payload.get("d_min"):
                        params.setdefault("d_min", vm.payload.get("d_min"))
                    if stage_id != "reduce":
                        if params.get("wavelength_range"):
                            wl_r = params["wavelength_range"]
                            vm.payload["wavelength_range"] = [
                                float(wl_r[0]),
                                float(wl_r[1]),
                            ]
                            vm.payload_kwargs["wavelength_range"] = (
                                float(wl_r[0]),
                                float(wl_r[1]),
                            )
                        if params.get("d_min"):
                            vm.payload["d_min"] = float(params["d_min"])
                            vm.payload_kwargs["d_min"] = float(params["d_min"])
                    cur_imgs = vm.payload_kwargs.get("images_h5") or vm.payload.get(
                        "images_h5"
                    )
                    if not cur_imgs and vm.replay_h5:
                        cur_imgs = (
                            vm.pipeline_runner._discover_or_fetch_galaxy_images_h5(
                                vm.replay_h5
                            )
                        )
                    if cur_imgs:
                        params.setdefault("images_file", str(cur_imgs))
                    if vm.replay_h5:
                        params.setdefault("peaks_file", str(vm.replay_h5))

                    def _on_complete(new_h5_path: str) -> None:
                        with vm_lock:
                            prev_replay_h5 = vm.replay_h5
                            prev_kwargs = dict(vm.payload_kwargs)
                            prev_mtz_path = vm.mtz_overlay_path
                            prev_mtz_info = vm.mtz_overlay_info
                            new_base_low = os.path.basename(new_h5_path).lower()
                            is_reduce_out = stage_id == "reduce" or (
                                "merged" in new_base_low
                                and "careless" not in new_base_low
                            )
                            if is_reduce_out:
                                vm.mtz_overlay_path = None
                                vm.mtz_overlay_info = None
                                vm.payload_kwargs["images_h5"] = new_h5_path
                                vm.payload_kwargs["frame_index"] = 0
                                vm.payload_kwargs.pop("wavelength_range", None)
                                vm.payload_kwargs.pop("d_min", None)
                            else:
                                vm.payload_kwargs["frame_index"] = int(
                                    vm.pipeline_runner.current_frame_index or 0
                                )
                                if params.get("d_min"):
                                    vm.payload_kwargs["d_min"] = float(params["d_min"])
                            vm.replay_h5 = new_h5_path
                            try:
                                vm.reload()
                            except Exception:
                                vm.replay_h5 = prev_replay_h5
                                vm.payload_kwargs = prev_kwargs
                                vm.mtz_overlay_path = prev_mtz_path
                                vm.mtz_overlay_info = prev_mtz_info
                                raise
                            if stage_id != "reduce":
                                if params.get("wavelength_range"):
                                    wl_r = params["wavelength_range"]
                                    vm.payload["wavelength_range"] = [
                                        float(wl_r[0]),
                                        float(wl_r[1]),
                                    ]
                                if params.get("d_min"):
                                    vm.payload["d_min"] = float(params["d_min"])
                                    vm.payload_kwargs["d_min"] = float(params["d_min"])

                    result = vm.pipeline_runner.submit_galaxy_stage(
                        stage_id, params=params, on_complete_h5=_on_complete
                    )
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"status": "error", "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if req_path == "/api/pipeline/upload_h5":
                import email.parser

                content_len = int(self.headers.get("Content-Length", "0") or 0)
                ctype = self.headers.get("Content-Type", "")
                raw_body = self.rfile.read(content_len) if content_len > 0 else b""
                try:
                    if vm.replay_h5 and os.path.isfile(vm.replay_h5):
                        fb_dir = os.path.dirname(os.path.abspath(vm.replay_h5))
                    else:
                        fb_dir = os.path.join(
                            tempfile.gettempdir(), f"subhkl_upload_{int(time.time())}"
                        )
                        os.makedirs(fb_dir, exist_ok=True)
                        register_temp_artifact(fb_dir)
                    saved_paths: list[str] = []
                    header_bytes = f"Content-Type: {ctype}\r\n\r\n".encode()
                    msg = email.parser.BytesParser().parsebytes(header_bytes + raw_body)
                    for part in msg.walk():
                        fname = part.get_filename()
                        if not fname:
                            continue
                        safe_name = os.path.basename(fname)
                        payload_bytes = part.get_payload(decode=True)
                        if payload_bytes is None:
                            continue
                        dest_path = os.path.join(fb_dir, safe_name)
                        with open(dest_path, "wb") as f_out:
                            f_out.write(payload_bytes)
                        register_temp_artifact(dest_path)
                        saved_paths.append(dest_path)
                    if len(saved_paths) == 1 and not saved_paths[0].lower().endswith(
                        (".nxs.h5", ".nxs", ".mtz")
                    ):
                        sp = saved_paths[0]
                        vm.mtz_overlay_path = None
                        vm.mtz_overlay_info = None
                        if "merged" in os.path.basename(sp).lower():
                            vm.payload_kwargs["images_h5"] = sp
                        vm.replay_h5 = sp
                        vm.reload()
                    result = {"status": "ok", "saved_paths": saved_paths}
                    code = 200
                except _NOVA_RECOVERABLE_ERRORS as exc:
                    result = {"status": "error", "error": str(exc)}
                    code = 400
                body = json.dumps(result).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    class _ThreadingServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    install_exit_cleanup_signal_handlers(vm.pipeline_runner.shutdown_cleanup)
    with _ThreadingServer((host, int(port)), _Handler) as httpd:
        print(
            f"Rainbow Zones server listening on http://{host}:{port}{ep_path or '/'} "
            f"(mode={default_mode})"
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            with contextlib.suppress(Exception):
                vm.pipeline_runner.shutdown_cleanup()
            cleanup_registered_temp_artifacts()
