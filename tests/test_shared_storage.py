"""Tests for three-tier shared storage orchestration and heterogeneous mount-point resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from subhkl.io.parser import app as cli_app
from subhkl.io.shared_storage import (
    allocate_shared_artifact_path,
    cleanup_shared_artifact,
    get_shared_dir_local,
    is_shared_pointer_file,
    parse_mountinfo,
    publish_shared_artifact,
    read_shared_pointer,
    resolve_input_h5_path,
    resolve_shared_local_path,
    to_shared_network_path,
    write_shared_pointer,
)
from subhkl.viz.nova_rainbow_view import (
    NovaGalaxyPipelineRunner,
    cleanup_registered_temp_artifacts,
    serve_rainbow_zones,
)
from subhkl.viz.rainbow_zones import validate_subhkl_ingestible_dataset

# Realistic /proc/self/mountinfo snapshots from ORNL runner hosts & containers
MOUNTINFO_IMAGINEX_NFS3 = """\
1590 1665 259:2 / / rw,relatime - xfs /dev/nvme0n1p2 rw,attr2,inode64
1591 1665 9:127 /pulsar_data /data rw,relatime - xfs /dev/md127 rw,attr2,inode64
1594 1665 0:53 / /scratch rw,relatime - nfs neutrondata.ornl.gov:/gpfs/neutronsfs/scratch rw,vers=3,proto=tcp
1595 1665 0:54 / /SNS ro,relatime - nfs snsdata.ornl.gov:/SNS ro,vers=3
"""

MOUNTINFO_IMAGINEX_NFSV4_ALIAS = """\
200 180 0:61 / /scratch rw,relatime - nfs4 neutrondata.ornl.gov:/scratch rw,vers=4.1
"""

MOUNTINFO_CONTAINER_SUBDIR_BIND = """\
310 290 0:53 /subhkl_shared /mnt/runner_shared rw,relatime - nfs neutrondata.ornl.gov:/gpfs/neutronsfs/scratch rw,vers=3
"""

MOUNTINFO_ANALYSIS_SNS_GPFS = """\
88 61 0:44 / /gpfs/neutronsfs rw,relatime - gpfs /dev/neutronsfs rw
"""

MOUNTINFO_ANALYSIS_SNS_NFS = """\
89 61 0:45 / /gpfs/neutronsfs rw,relatime - nfs neutrondata.ornl.gov:/gpfs/neutronsfs rw,vers=3
"""


def test_mountinfo_translation_across_imaginex_and_analysis_hosts() -> None:
    """Verify full network path <-> local mount-point resolution across imaginex-{1,2}, NFSv4, subdir bind, and analysis.sns.gov."""
    entries_ix = parse_mountinfo(MOUNTINFO_IMAGINEX_NFS3)
    assert any(
        e.mount_point == "/scratch"
        and e.mount_source == "neutrondata.ornl.gov:/gpfs/neutronsfs/scratch"
        for e in entries_ix
    )

    # 1. File produced on imaginex-1 at /scratch/subhkl_shared/run_a_merged.h5 -> full network path
    net_from_ix = to_shared_network_path(
        "/scratch/subhkl_shared/run_a_merged.h5",
        mountinfo_text=MOUNTINFO_IMAGINEX_NFS3,
    )
    assert (
        net_from_ix
        == "neutrondata.ornl.gov:/gpfs/neutronsfs/scratch/subhkl_shared/run_a_merged.h5"
    )

    # Same full network path produced even when host uses NFSv4 pseudo-root neutrondata.ornl.gov:/scratch
    net_from_v4 = to_shared_network_path(
        "/scratch/subhkl_shared/run_a_merged.h5",
        mountinfo_text=MOUNTINFO_IMAGINEX_NFSV4_ALIAS,
    )
    assert net_from_v4 == net_from_ix

    # Same full network path produced inside a container bind-mounting /scratch/subhkl_shared at /mnt/runner_shared
    net_from_bind = to_shared_network_path(
        "/mnt/runner_shared/run_a_merged.h5",
        mountinfo_text=MOUNTINFO_CONTAINER_SUBDIR_BIND,
    )
    assert net_from_bind == net_from_ix

    # 2. Resolve that full network path on every runner host topology:
    assert (
        resolve_shared_local_path(
            net_from_ix,
            mountinfo_text=MOUNTINFO_IMAGINEX_NFS3,
            must_exist=False,
        )
        == "/scratch/subhkl_shared/run_a_merged.h5"
    )
    assert (
        resolve_shared_local_path(
            net_from_ix,
            mountinfo_text=MOUNTINFO_IMAGINEX_NFSV4_ALIAS,
            must_exist=False,
        )
        == "/scratch/subhkl_shared/run_a_merged.h5"
    )
    assert (
        resolve_shared_local_path(
            net_from_ix,
            mountinfo_text=MOUNTINFO_CONTAINER_SUBDIR_BIND,
            must_exist=False,
        )
        == "/mnt/runner_shared/run_a_merged.h5"
    )
    assert (
        resolve_shared_local_path(
            net_from_ix,
            mountinfo_text=MOUNTINFO_ANALYSIS_SNS_GPFS,
            must_exist=False,
        )
        == "/gpfs/neutronsfs/scratch/subhkl_shared/run_a_merged.h5"
    )
    assert (
        resolve_shared_local_path(
            net_from_ix,
            mountinfo_text=MOUNTINFO_ANALYSIS_SNS_NFS,
            must_exist=False,
        )
        == "/gpfs/neutronsfs/scratch/subhkl_shared/run_a_merged.h5"
    )

    # 3. Single canonical SUBHKL_SHARED_DIR=/gpfs/neutronsfs/scratch/subhkl_shared resolves locally on EVERY host!
    canon_cfg = "/gpfs/neutronsfs/scratch/subhkl_shared"
    assert (
        get_shared_dir_local(
            require=True,
            create=False,
            shared_dir_override=canon_cfg,
            mountinfo_text=MOUNTINFO_IMAGINEX_NFS3,
        )
        == "/scratch/subhkl_shared"
    )
    assert (
        get_shared_dir_local(
            require=True,
            create=False,
            shared_dir_override=canon_cfg,
            mountinfo_text=MOUNTINFO_ANALYSIS_SNS_GPFS,
        )
        == "/gpfs/neutronsfs/scratch/subhkl_shared"
    )
    assert (
        get_shared_dir_local(
            require=True,
            create=False,
            shared_dir_override=canon_cfg,
            mountinfo_text=MOUNTINFO_CONTAINER_SUBDIR_BIND,
        )
        == "/mnt/runner_shared"
    )


def test_mandatory_subhkl_shared_dir_in_galaxy_exec_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify SUBHKL_SHARED_DIR is strictly mandatory in Galaxy execution mode, and optional in local mode."""
    monkeypatch.delenv("SUBHKL_SHARED_DIR", raising=False)
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "galaxy")
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(tmp_path / "nvme_local"))

    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://ndip.ornl.gov",
        galaxy_api_key="fake_key",
        history_id="hist_1",
    )
    assert runner.is_explicit_galaxy_exec_mode() is True
    assert runner.is_galaxy_exec_mode() is True

    # Missing SUBHKL_SHARED_DIR fails immediately on stage submission and server startup
    with pytest.raises(RuntimeError, match="SUBHKL_SHARED_DIR"):
        runner.submit_galaxy_stage("reduce", {"runs": "1001"})

    with pytest.raises(RuntimeError, match="SUBHKL_SHARED_DIR"):
        serve_rainbow_zones(payload={"panels": [], "peaks": [], "zones": []})

    # CLI merge-images --shared-pointer-out also enforces mandatory SUBHKL_SHARED_DIR
    cli = CliRunner()
    res_missing = cli.invoke(
        cli_app,
        [
            "merge-images",
            str(tmp_path / "*.h5"),
            "auto",
            "10",
            "10",
            "10",
            "90",
            "90",
            "90",
            "P 1",
            "--shared-pointer-out",
            str(tmp_path / "scan_ref.json"),
        ],
    )
    assert res_missing.exit_code != 0
    assert "SUBHKL_SHARED_DIR" in (
        str(res_missing.exception or "") + res_missing.output
    )

    # In SUBHKL_EXEC_MODE=local, SUBHKL_SHARED_DIR is NOT required
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "local")
    assert runner.is_galaxy_exec_mode() is False


def test_cli_pointer_staging_cross_mount_deref_provenance_and_ephemeral_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI merge-images, solve, peak-predictor, integrator, mtz-exporter, and cleanup-shared with heterogeneous mounts."""
    # Simulate two different runner hosts mounting the SAME underlying physical shared volume at different paths:
    # Host A (imaginex-1): mounts neutrondata.ornl.gov:/gpfs/neutronsfs/scratch at <gpfs_root>/scratch
    # Host B (analysis.sns.gov): mounts /dev/neutronsfs (/gpfs/neutronsfs) at <gpfs_root> (with scratch/ inside it)
    nvme_a = tmp_path / "host_a_nvme"
    nvme_b = tmp_path / "host_b_nvme"
    gpfs_root = tmp_path / "gpfs_root"
    phys_shared = gpfs_root / "scratch" / "subhkl_shared"
    nvme_a.mkdir()
    nvme_b.mkdir()
    phys_shared.mkdir(parents=True)

    mount_a = f"10 1 0:53 / {gpfs_root / 'scratch'} rw - nfs neutrondata.ornl.gov:/gpfs/neutronsfs/scratch rw\n"
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_a))
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(phys_shared))

    # Create two synthetic step-local *.reduce.h5 files on Host A's local NVMe
    for idx_r in (0, 1):
        r_p = nvme_a / f"CG4D_262{idx_r}.reduce.h5"
        with h5py.File(r_p, "w") as hf:
            hf.create_dataset(
                "images", data=np.ones((1, 4, 4), dtype=np.float32) * (idx_r + 1)
            )
            hf.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
            hf.create_group("goniometer").create_dataset(
                "R", data=np.eye(3, dtype=np.float64)
            )
            hf["goniometer"].create_dataset(
                "angles", data=np.array([float(idx_r * 5.0)], dtype=np.float64)
            )
            hf["goniometer"].create_dataset(
                "axes", data=np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
            )
            hf["goniometer"].create_dataset("names", data=np.array([b"omega"]))
            hf.create_group("instrument").create_dataset(
                "wavelength", data=np.array([2.0, 4.5], dtype=np.float64)
            )

    cli = CliRunner()
    scan_ptr_file = tmp_path / "galaxy_hda_scan_ref.json"
    raw_src_0 = "/HFIR/CG4D/IPTS-37331/nexus/CG4D_2620.nxs.h5"
    raw_src_1 = "/HFIR/CG4D/IPTS-37331/nexus/CG4D_2621.nxs.h5"

    res_merge = cli.invoke(
        cli_app,
        [
            "merge-images",
            f"{nvme_a}/CG4D_262*.reduce.h5",
            "auto",
            "54.0",
            "58.0",
            "62.0",
            "90.0",
            "90.0",
            "90.0",
            "P 21 21 21",
            "--shared-pointer-out",
            str(scan_ptr_file),
            "--instrument",
            "CG4D",
            "--source-file",
            raw_src_0,
            "--source-file",
            raw_src_1,
        ],
    )
    assert res_merge.exit_code == 0, res_merge.output
    assert is_shared_pointer_file(scan_ptr_file)
    assert scan_ptr_file.stat().st_size < 1024  # tiny pointer HDA (<1 KB)

    ptr_doc = read_shared_pointer(scan_ptr_file)
    assert ptr_doc is not None
    assert ptr_doc["delete_after_use"] is False
    shared_files = list(phys_shared.glob("subhkl_shared_*_merged.h5"))
    assert len(shared_files) == 1
    shared_merged_local = shared_files[0]

    # Rewrite pointer's network_path using Host A's NFS export and verify Host B resolves it locally!
    net_url_a = to_shared_network_path(shared_merged_local, mountinfo_text=mount_a)
    assert net_url_a.startswith(
        "neutrondata.ornl.gov:/gpfs/neutronsfs/scratch/subhkl_shared/"
    )
    raw_ptr_json = json.loads(scan_ptr_file.read_text(encoding="utf-8"))
    raw_ptr_json["network_path"] = net_url_a
    scan_ptr_file.write_text(json.dumps(raw_ptr_json), encoding="utf-8")

    # Host B mounts /dev/neutronsfs (/gpfs/neutronsfs) at gpfs_root (different mount depth, scratch implicitly inside!)
    mount_b = f"22 1 0:99 / {gpfs_root} rw - gpfs /dev/neutronsfs rw\n"
    resolved_on_b, _ = resolve_input_h5_path(
        scan_ptr_file,
        mountinfo_text=mount_b,
        must_exist=True,
    )
    assert Path(resolved_on_b) == shared_merged_local

    # Switch active env to Host B (both SUBHKL_MOUNTINFO_PATH and NVMe scratch) and run `solve`,
    # `peak-predictor --shared-pointer-out`, `integrator`, `mtz-exporter`, and `cleanup-shared --if-ephemeral`!
    mount_b_file = tmp_path / "host_b_mountinfo"
    mount_b_file.write_text(mount_b, encoding="utf-8")
    monkeypatch.setenv("SUBHKL_MOUNTINFO_PATH", str(mount_b_file))
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_b))
    solve_out = tmp_path / "thin_solve.h5"

    def _fake_band_solve(frames_fn: str, out_fn: str, **_kw: object) -> dict[str, str]:
        assert Path(frames_fn) == shared_merged_local
        with h5py.File(out_fn, "w") as sf:
            sf.create_group("sample").create_dataset(
                "U", data=np.eye(3, dtype=np.float64)
            )
            sf["sample"].create_dataset("B", data=np.eye(3, dtype=np.float64) / 54.0)
            sf["sample"].create_dataset(
                "cell", data=np.array([54, 58, 62, 90, 90, 90], dtype=np.float64)
            )
        return {"status": "converged"}

    with patch("subhkl.scan_refine.run_scan_refinement", side_effect=_fake_band_solve):
        res_solve = cli.invoke(
            cli_app,
            [
                "solve",
                str(scan_ptr_file),
                str(solve_out),
                "--instrument",
                "CG4D",
                "--refinement",
                "band",
            ],
        )
    assert res_solve.exit_code == 0, res_solve.output
    # Thin solve.h5 MUST carry full source_files provenance for History restarts!
    with h5py.File(solve_out, "r") as sf_chk:
        stamped_srcs = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in sf_chk["source_files"][()]
        ]
        assert stamped_srcs == [raw_src_0, raw_src_1]
        assert str(sf_chk.attrs["instrument"]) == "CG4D"

    # Stage ephemeral peak_predictor.h5 via --shared-pointer-out
    pred_ptr_file = tmp_path / "galaxy_hda_pred_ref.json"

    def _fake_predictor(
        scan_fn: str, _inst: str, idx_fn: str, out_fn: str, **_kw: object
    ) -> None:
        assert Path(scan_fn) == shared_merged_local
        assert Path(idx_fn) == solve_out
        with h5py.File(out_fn, "w") as pf:
            pf.create_group("peaks").create_dataset(
                "hkl", data=np.array([[1, 0, 0]], dtype=np.int32)
            )

    with patch("subhkl.io.parser.run_peak_predictor", side_effect=_fake_predictor):
        res_pred = cli.invoke(
            cli_app,
            [
                "peak-predictor",
                str(scan_ptr_file),
                "CG4D",
                str(solve_out),
                "unused_out.h5",
                "--shared-pointer-out",
                str(pred_ptr_file),
            ],
        )
    assert res_pred.exit_code == 0, res_pred.output
    pred_ptr_doc = read_shared_pointer(pred_ptr_file)
    assert pred_ptr_doc is not None
    assert pred_ptr_doc["delete_after_use"] is True
    pred_shared_files = list(phys_shared.glob("subhkl_shared_*_peak_predictor.h5"))
    assert len(pred_shared_files) == 1
    shared_pred_local = pred_shared_files[0]

    # Run integrator + mtz-exporter consuming BOTH tiny JSON pointers
    integ_out = tmp_path / "thin_integrator.h5"
    mtz_out = tmp_path / "thin_reflections.mtz"

    def _fake_integrator(**kw: object) -> None:
        assert Path(str(kw["filename"])) == shared_merged_local
        assert Path(str(kw["integration_peaks_filename"])) == shared_pred_local
        with h5py.File(str(kw["output_filename"]), "w") as inf:
            inf.create_group("peaks").create_dataset(
                "intensity", data=np.array([42.0], dtype=np.float32)
            )

    def _fake_mtz(
        idx_fn: str,
        out_mtz: str,
        _sg: object,
        predictions_file: str | None = None,
        **_kw: object,
    ) -> None:
        assert Path(idx_fn) == integ_out
        assert Path(str(predictions_file)) == shared_pred_local
        Path(out_mtz).write_bytes(b"MTZ_THIN")

    with (
        patch("subhkl.io.parser.run_rbf_integrator", side_effect=_fake_integrator),
        patch("subhkl.io.parser.run_mtz_exporter", side_effect=_fake_mtz),
    ):
        assert (
            cli.invoke(
                cli_app,
                [
                    "integrator",
                    str(scan_ptr_file),
                    "CG4D",
                    str(pred_ptr_file),
                    str(integ_out),
                ],
            ).exit_code
            == 0
        )
        assert (
            cli.invoke(
                cli_app,
                [
                    "mtz-exporter",
                    str(integ_out),
                    str(mtz_out),
                    "--predictions-file",
                    str(pred_ptr_file),
                ],
            ).exit_code
            == 0
        )

    with h5py.File(integ_out, "r") as inf_chk:
        assert len(inf_chk["source_files"]) == 2

    # cleanup-shared --if-ephemeral deletes peak_predictor.h5 immediately, while keeping merged.h5!
    res_clean = cli.invoke(
        cli_app,
        [
            "cleanup-shared",
            str(scan_ptr_file),
            str(pred_ptr_file),
            "--if-ephemeral",
        ],
    )
    assert res_clean.exit_code == 0
    assert not shared_pred_local.exists()
    assert shared_merged_local.exists()

    # Once peak_predictor.h5 is deleted, attempting to ingest its stale pointer raises a clean ValueError
    with pytest.raises(ValueError, match="cleaned up"):
        validate_subhkl_ingestible_dataset(pred_ptr_file)

    # Explicit non-ephemeral cleanup removes merged.h5 as well
    assert (
        cleanup_shared_artifact(scan_ptr_file, only_if_delete_after_use=False) is True
    )
    assert not shared_merged_local.exists()


def test_galaxy_runner_pointer_orchestration_and_shared_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Reduce -> Solve -> Integrate pointer handoff (0 multi-GB uploads/copies), dataset-switch/exit cleanup, and Galaxy hydration."""
    nvme_dir = tmp_path / "nvme_staging"
    shared_dir = tmp_path / "shared_nfs"
    nvme_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "galaxy")
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_dir))
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(shared_dir))

    runner = NovaGalaxyPipelineRunner(
        galaxy_url="http://localhost:8080",
        galaxy_api_key="fake_key",
        history_id="hist_gx",
    )

    # Simulate Prepare Peaks runner writing merged.h5 in SUBHKL_SHARED_DIR + tiny scan_ref.json HDA
    shared_merged_1, _ = allocate_shared_artifact_path("merged.h5", tag="CG4D_2625")
    with h5py.File(shared_merged_1, "w") as mf:
        mf.attrs["instrument"] = "CG4D"
        mf.create_dataset("images", data=np.ones((1, 8, 8), dtype=np.float32))
        mf.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
        mf.create_dataset(
            "source_files",
            data=np.array(
                [b"/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"], dtype=object
            ),
        )
        mf.create_dataset("files", data=np.array([b"CG4D_2625.nxs.h5"]))
    scan_ptr_hda = tmp_path / "hda_scan_1.json"
    write_shared_pointer(scan_ptr_hda, shared_merged_1, artifact="merged.h5")

    shared_pred_1, _ = allocate_shared_artifact_path("peak_predictor.h5", tag="CG4D")
    with h5py.File(shared_pred_1, "w") as pf:
        pf.create_group("peaks").create_dataset(
            "hkl", data=np.array([[1, 0, 0]], dtype=np.int32)
        )
    pred_ptr_hda = tmp_path / "hda_pred_1.json"
    write_shared_pointer(
        pred_ptr_hda,
        shared_pred_1,
        artifact="peak_predictor.h5",
        delete_after_use=True,
    )

    thin_solve_hda = tmp_path / "hda_solve_1.h5"
    with h5py.File(thin_solve_hda, "w") as sf:
        sf.attrs["instrument"] = "CG4D"
        sf.create_dataset(
            "source_files",
            data=[b"/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"],
        )
        sf.create_group("sample").create_dataset("U", data=np.eye(3))
        sf["sample"].create_dataset("B", data=np.eye(3) / 50.0)
        sf["sample"].create_dataset(
            "cell", data=np.array([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
        )
        sf.create_group("instrument").create_dataset(
            "wavelength", data=np.array([2.0, 4.5])
        )

    thin_integ_hda = tmp_path / "hda_integ_1.h5"
    with h5py.File(thin_integ_hda, "w") as inf:
        inf.attrs["instrument"] = "CG4D"
        inf.create_group("peaks").create_dataset("intensity", data=np.array([10.0]))

    hda_blobs = {
        "ds_scan_ptr_1": scan_ptr_hda,
        "ds_solve_1": thin_solve_hda,
        "ds_pred_ptr_1": pred_ptr_hda,
        "ds_integ_1": thin_integ_hda,
    }
    purged_ids: list[str] = []
    uploaded_files: list[tuple[str, str, int]] = []
    dispatched_tools: list[tuple[str, dict[str, object]]] = []

    def _fake_api(
        method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        if method == "PUT" and isinstance(payload, dict) and payload.get("deleted"):
            purged_ids.append(path.rsplit("/", 1)[-1])
            return {"deleted": True}
        if "/api/jobs/" in path:
            return {
                "state": "ok",
                "exit_code": 0,
                "tool_stdout": "done\n",
                "tool_stderr": "",
            }
        if "/api/datasets/" in path:
            did = path.rsplit("/", 1)[-1].split("?")[0]
            return {"id": did, "name": f"{did}.json", "state": "ok", "deleted": False}
        return {}

    def _fake_dl(ds_id: str, dest: str, **_kw: object) -> int:
        src_p = hda_blobs[ds_id]
        data = src_p.read_bytes()
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        Path(dest).write_bytes(data)
        return len(data)

    class _FakeTools:
        def upload_file(self, path: str, file_name: str, file_type: str, **_kw: object):
            uploaded_files.append((file_name, file_type, os.path.getsize(path)))
            return {"outputs": [{"id": f"up_{len(uploaded_files)}"}]}

        def run_tool(
            self, history_id: str, tool_id: str, tool_inputs: dict[str, object]
        ):
            dispatched_tools.append((tool_id, dict(tool_inputs)))
            if tool_id == "imaginex_subhkl_prepare_peaks":
                return {
                    "jobs": [{"id": "j_red"}],
                    "outputs": [
                        {
                            "id": "ds_scan_ptr_1",
                            "name": "scan.ref.json",
                            "file_ext": "json",
                        }
                    ],
                }
            if tool_id == "imaginex_subhkl_solve":
                return {
                    "jobs": [{"id": "j_sol"}],
                    "outputs": [
                        {"id": "ds_solve_1", "name": "solve.h5", "file_ext": "h5"}
                    ],
                }
            if tool_id == "imaginex_subhkl_predict":
                return {
                    "jobs": [{"id": "j_pred"}],
                    "outputs": [
                        {
                            "id": "ds_pred_ptr_1",
                            "name": "peak_predictor.ref.json",
                            "file_ext": "json",
                        }
                    ],
                }
            if tool_id == "imaginex_subhkl_integrate":
                return {
                    "jobs": [{"id": "j_int"}],
                    "outputs": [
                        {
                            "id": "ds_integ_1",
                            "name": "integrator.h5",
                            "file_ext": "h5",
                        }
                    ],
                }
            raise AssertionError(f"Unexpected tool {tool_id}")

    class _FakeDatasets:
        def show_dataset(self, ds_id: str):
            return {"id": ds_id, "name": f"{ds_id}", "state": "ok", "deleted": False}

        def get_datasets(self, **_kw: object):
            return []

        def wait_for_dataset(self, *_a: object, **_kw: object):
            return None

    fake_store = SimpleNamespace(
        history_id="hist_gx",
        nova_connection=SimpleNamespace(
            galaxy_instance=SimpleNamespace(
                base_url="http://localhost:8080",
                tools=_FakeTools(),
                datasets=_FakeDatasets(),
            )
        ),
    )

    monkeypatch.setattr(runner, "_api_request", _fake_api)
    monkeypatch.setattr(runner, "_download_dataset_to_file", _fake_dl)
    monkeypatch.setattr(runner, "get_datastore", lambda *_a, **_k: fake_store)
    monkeypatch.setattr(runner, "_has_tool_on_galaxy", lambda _t: True)
    monkeypatch.setattr(runner, "_get_tool_version", lambda _t: "0.3.0")
    monkeypatch.setattr(
        runner, "attach_dataset_cell_metadata_from_file", lambda *_a, **_k: None
    )

    # 1. Step 1 (Reduce): receives ~300B pointer, creates zero-copy symlink in local NVMe work_dir
    rec_red: dict[str, object] = {"job_id": "r1", "stage_id": "reduce", "stdout": ""}
    runner._dispatch_nova_galaxy_thread(
        "reduce",
        {
            "instrument": "CG4D",
            "nexus_files": ["/HFIR/CG4D/IPTS-37331/nexus/CG4D_2625.nxs.h5"],
            "cell": [50, 60, 70, 90, 90, 90],
            "space_group": "P 21 21 21",
        },
        rec_red,
    )
    out_merged = str(rec_red["output_h5_path"])
    assert os.path.islink(out_merged)  # zero-copy symlink into SUBHKL_SHARED_DIR!
    assert os.path.realpath(out_merged) == os.path.abspath(shared_merged_1)

    # 2. Step 2 (Solve): reuses existing pointer HDA ds_scan_ptr_1 with ZERO uploads!
    rec_sol: dict[str, object] = {"job_id": "s1", "stage_id": "solve", "stdout": ""}
    runner._dispatch_nova_galaxy_thread(
        "solve",
        {
            "instrument": "CG4D",
            "images_file": out_merged,
            "cell": [50, 60, 70, 90, 90, 90],
            "space_group": "P 21 21 21",
        },
        rec_sol,
    )
    assert uploaded_files == []  # 0 bytes uploaded for Solve!
    assert dispatched_tools[-1][0] == "imaginex_subhkl_solve"
    assert dispatched_tools[-1][1]["scan"] == {"src": "hda", "id": "ds_scan_ptr_1"}
    staged_solve = str(rec_sol["output_h5_path"])

    # 3. Step 3 (Integrate = Predict + Integrate):
    #    - Reuses ds_scan_ptr_1 for scan (no multi-GB upload!)
    #    - Ephemeral shared_pred_1 is DELETED immediately after Integrate finishes, while shared_merged_1 stays!
    rec_int: dict[str, object] = {"job_id": "i1", "stage_id": "integrate", "stdout": ""}
    runner._dispatch_nova_galaxy_thread(
        "integrate",
        {
            "instrument": "CG4D",
            "images_file": out_merged,
            "peaks_file": staged_solve,
            "cell": [50, 60, 70, 90, 90, 90],
            "space_group": "P 21 21 21",
        },
        rec_int,
    )
    assert not os.path.exists(shared_pred_1)
    assert "ds_pred_ptr_1" in purged_ids
    assert os.path.exists(shared_merged_1)

    # 4. Restarting from thin solve.h5 in SUBHKL_EXEC_MODE=galaxy triggers Galaxy runner hydration (NOT local GPU!)
    #    Use a non-mounted /HFIR path so hydration is verified even when UI/CI hosts do NOT mount /HFIR or /SNS.
    other_solve = tmp_path / "restart_solve.h5"
    other_solve.write_bytes(thin_solve_hda.read_bytes())
    with h5py.File(other_solve, "a") as osf:
        del osf["source_files"]
        osf.create_dataset(
            "source_files",
            data=[b"/HFIR/CG4D/IPTS-999999/nexus/CG4D_999999.nxs.h5"],
        )
    assert not os.path.exists("/HFIR/CG4D/IPTS-999999/nexus/CG4D_999999.nxs.h5")
    with patch.object(
        runner,
        "submit_galaxy_stage",
        return_value={"job": {"job_id": "hyd_gx_1", "state": "queued"}},
    ) as mock_sub_gx:
        hyd_info = runner._maybe_trigger_hydrate_reduce(
            MagicMock(), MagicMock(), str(other_solve)
        )
    assert hyd_info is not None
    assert hyd_info["job_id"] == "hyd_gx_1"
    assert hyd_info["hydrate_only"] is True
    mock_sub_gx.assert_called_once()
    assert mock_sub_gx.call_args.args[0] == "reduce"
    assert mock_sub_gx.call_args.args[1]["hydrate_only"] is True
    assert mock_sub_gx.call_args.args[1]["nexus_files"] == [
        "/HFIR/CG4D/IPTS-999999/nexus/CG4D_999999.nxs.h5"
    ]

    # 5. Switching to a different dataset work_dir cleans up shared_merged_1 and purges ds_scan_ptr_1!
    removed_on_switch = runner.release_shared_scan_intermediates(
        keep_work_dir=str(nvme_dir / "subhkl_work_OTHER_DATASET"),
        keep_paths=set(),
    )
    assert os.path.abspath(shared_merged_1) in removed_on_switch
    assert not os.path.exists(shared_merged_1)
    assert "ds_scan_ptr_1" in purged_ids

    # 6. Process-exit cleanup also deletes any remaining registered shared intermediate
    shared_merged_2, _ = allocate_shared_artifact_path("merged.h5", tag="EXIT_TEST")
    Path(shared_merged_2).write_bytes(b"TMP_SHARED")
    runner._resolve_work_dir_and_merged(shared_merged_2)
    with h5py.File(shared_merged_2, "w") as mf2:
        mf2.create_dataset("images", data=np.zeros((1, 2, 2), dtype=np.float32))
    runner._resolve_work_dir_and_merged(shared_merged_2)
    assert os.path.exists(shared_merged_2)
    cleaned_exit = cleanup_registered_temp_artifacts()
    assert os.path.abspath(shared_merged_2) in cleaned_exit
    assert not os.path.exists(shared_merged_2)


def test_nfs_race_free_uuid_staging_and_fsync_before_pointer_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify full 128-bit UUIDv4 filenames (cold NFS cache miss), zero pre-create stat, and os.fsync+close BEFORE JSON pointer publish."""
    import builtins
    import re

    nvme_dir = tmp_path / "nvme_local"
    shared_dir = tmp_path / "nfs_shared"
    nvme_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_dir))
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(shared_dir))

    src_h5 = nvme_dir / "local_scan.h5"
    with h5py.File(src_h5, "w") as hf:
        hf.create_dataset(
            "images", data=np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        )

    ptr_out = tmp_path / "out" / "scan_ref.json"

    events: list[tuple[str, str]] = []
    fd_to_tag: dict[int, str] = {}
    orig_os_open = os.open
    orig_os_fsync = os.fsync
    orig_os_close = os.close
    orig_os_stat = os.stat
    orig_open = builtins.open

    def _tracked_stat(path: object, *args: object, **kwargs: object):
        p_str = os.fspath(path) if isinstance(path, (str, os.PathLike)) else ""
        if isinstance(p_str, str) and "subhkl_shared_" in os.path.basename(p_str):
            events.append(("stat_shared", os.path.basename(p_str)))
        return orig_os_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    def _tracked_os_open(
        path: str, flags: int, mode: int = 0o777, **kwargs: object
    ) -> int:
        fd = orig_os_open(path, flags, mode, **kwargs)
        p_str = os.fspath(path)
        bn = os.path.basename(p_str)
        if bn.startswith("subhkl_shared_"):
            assert flags & os.O_CREAT and flags & os.O_EXCL
            fd_to_tag[fd] = "shared_h5"
            events.append(("open_shared_h5", bn))
        elif os.path.abspath(p_str) == os.path.abspath(shared_dir):
            fd_to_tag[fd] = "shared_dir"
        elif os.path.abspath(p_str) == os.path.abspath(ptr_out.parent):
            fd_to_tag[fd] = "ptr_dir"
        return fd

    def _tracked_builtin_open(
        file: object, mode: str = "r", *args: object, **kwargs: object
    ):
        fh = orig_open(file, mode, *args, **kwargs)  # type: ignore[call-overload]
        if isinstance(file, (str, os.PathLike)):
            p_str = os.fspath(file)
            if os.path.abspath(p_str) == os.path.abspath(ptr_out) and "w" in mode:
                fd_to_tag[fh.fileno()] = "ptr_json"
                events.append(("open_ptr_json", os.path.basename(p_str)))

                orig_fh_close = fh.close

                def _wrapped_fh_close() -> None:
                    fd_to_tag.pop(fh.fileno(), None)
                    orig_fh_close()
                    events.append(("close_ptr_json", os.path.basename(p_str)))

                fh.close = _wrapped_fh_close  # type: ignore[method-assign]
        return fh

    def _tracked_fsync(fd: int) -> None:
        tag = fd_to_tag.get(fd)
        if tag:
            events.append((f"fsync_{tag}", str(fd)))
        orig_os_fsync(fd)

    def _tracked_close(fd: int) -> None:
        tag = fd_to_tag.pop(fd, None)
        orig_os_close(fd)
        if tag:
            events.append((f"close_{tag}", str(fd)))

    monkeypatch.setattr(os, "stat", _tracked_stat)
    monkeypatch.setattr(os, "open", _tracked_os_open)
    monkeypatch.setattr(os, "fsync", _tracked_fsync)
    monkeypatch.setattr(os, "close", _tracked_close)
    monkeypatch.setattr(builtins, "open", _tracked_builtin_open)

    shared_1, doc_1 = publish_shared_artifact(
        src_h5,
        ptr_out,
        artifact="merged.h5",
        metadata={"instrument": "CG4D"},
    )
    ptr_out_2 = tmp_path / "out" / "scan_ref_2.json"
    shared_2, doc_2 = publish_shared_artifact(
        src_h5,
        ptr_out_2,
        artifact="merged.h5",
        metadata={"instrument": "CG4D"},
    )

    # 1. Distinct full 32-hex 128-bit UUIDv4 filenames on every publication
    m1 = re.match(
        r"^subhkl_shared_([0-9a-f]{32})_merged\.h5$", os.path.basename(shared_1)
    )
    m2 = re.match(
        r"^subhkl_shared_([0-9a-f]{32})_merged\.h5$", os.path.basename(shared_2)
    )
    assert m1 is not None and m2 is not None
    assert m1.group(1) != m2.group(1)
    assert doc_1["artifact_uuid"] == m1.group(1)
    assert doc_2["artifact_uuid"] == m2.group(1)
    assert doc_1["size_bytes"] == src_h5.stat().st_size

    # 2. Strict causal ordering for first publication:
    #    ZERO stat on shared filename before O_CREAT|O_EXCL open -> fsync(shared_h5) -> close(shared_h5) -> open(ptr_json) -> fsync(ptr_json) -> close(ptr_json)
    bn_1 = os.path.basename(shared_1)
    idx_open_h5 = next(
        i
        for i, (ev, arg) in enumerate(events)
        if ev == "open_shared_h5" and arg == bn_1
    )
    assert not any(
        ev == "stat_shared" and arg == bn_1 for ev, arg in events[:idx_open_h5]
    ), (
        f"Unexpected pre-creation stat on NFS target before O_CREAT|O_EXCL: {events[:idx_open_h5]}"
    )

    idx_fsync_h5 = next(
        i
        for i in range(idx_open_h5 + 1, len(events))
        if events[i][0] == "fsync_shared_h5"
    )
    idx_close_h5 = next(
        i
        for i in range(idx_fsync_h5 + 1, len(events))
        if events[i][0] == "close_shared_h5"
    )
    idx_open_ptr = next(i for i, (ev, _) in enumerate(events) if ev == "open_ptr_json")
    idx_fsync_ptr = next(
        i
        for i in range(idx_open_ptr + 1, len(events))
        if events[i][0] == "fsync_ptr_json"
    )
    idx_close_ptr = next(
        i
        for i in range(idx_fsync_ptr + 1, len(events))
        if events[i][0] == "close_ptr_json"
    )

    assert (
        idx_open_h5
        < idx_fsync_h5
        < idx_close_h5
        < idx_open_ptr
        < idx_fsync_ptr
        < idx_close_ptr
    )
    assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"


def test_readonly_ui_shared_dir_and_worker_cleanup_tool_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify UI requires only read-only (ro) SUBHKL_SHARED_DIR access and delegates cleanup to imaginex_subhkl_cleanup_shared."""
    nvme_dir = tmp_path / "ui_nvme"
    shared_dir = tmp_path / "shared_nfs_ro"
    nvme_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SUBHKL_EXEC_MODE", "galaxy")
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_dir))
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(shared_dir))

    # 1. Worker (with rw access) produces merged.h5 + scan.ref.json BEFORE UI mounts shared_dir read-only
    shared_merged_local, shared_merged_net = allocate_shared_artifact_path(
        "merged.h5", tag="RO_TEST"
    )
    with h5py.File(shared_merged_local, "w") as mf:
        mf.attrs["instrument"] = "CG4D"
        mf.create_dataset("images", data=np.ones((1, 4, 4), dtype=np.float32))
        mf.create_dataset("bank_ids", data=np.array([1], dtype=np.int32))
    scan_ptr_hda = tmp_path / "hda_scan_ro.json"
    write_shared_pointer(scan_ptr_hda, shared_merged_local, artifact="merged.h5")

    # 2. Lock shared_dir to read-only (0o555) to simulate a strict `ro` bind-mount in the UI container
    os.chmod(shared_dir, 0o555)
    try:
        runner = NovaGalaxyPipelineRunner(
            galaxy_url="http://localhost:8080",
            galaxy_api_key="fake_key",
            history_id="hist_ro",
        )

        # UI startup / stage check succeeds with read-only access (writable=False)
        assert runner.require_shared_dir_for_galaxy_mode() == str(shared_dir)

        # Worker-side writer check properly rejects read-only mount (writable=True)
        with pytest.raises(RuntimeError, match="not a writable directory"):
            get_shared_dir_local(require=True, create=False, writable=True)

        dispatched_cleanup: list[tuple[str, dict[str, object]]] = []
        purged_ids: list[str] = []

        def _fake_api(
            method: str,
            endpoint: str,
            payload: dict[str, object] | None = None,
            **_kw: object,
        ):
            if method == "PUT" and isinstance(payload, dict) and payload.get("deleted"):
                purged_ids.append(endpoint.rsplit("/", 1)[-1])
                return {"id": purged_ids[-1], "deleted": True}
            if "/api/jobs/" in endpoint:
                return {
                    "state": "ok",
                    "exit_code": 0,
                    "tool_stdout": "done\n",
                    "tool_stderr": "",
                }
            if "/api/datasets/" in endpoint:
                did = endpoint.rsplit("/", 1)[-1].split("?")[0]
                return {
                    "id": did,
                    "name": f"{did}.json",
                    "state": "ok",
                    "deleted": False,
                }
            return {}

        def _fake_dl(ds_id: str, dest: str, **_kw: object) -> int:
            assert ds_id == "ds_scan_ro_1"
            data = scan_ptr_hda.read_bytes()
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            Path(dest).write_bytes(data)
            return len(data)

        class _FakeTools:
            def run_tool(
                self,
                history_id: str,
                tool_id: str,
                tool_inputs: dict[str, object],
            ):
                if tool_id == "imaginex_subhkl_prepare_peaks":
                    return {
                        "jobs": [{"id": "j_prep_ro"}],
                        "outputs": [
                            {
                                "id": "ds_scan_ro_1",
                                "name": "scan_ref.json",
                                "file_ext": "json",
                            }
                        ],
                    }
                if tool_id == "imaginex_subhkl_cleanup_shared":
                    dispatched_cleanup.append((tool_id, dict(tool_inputs)))
                    # Simulate worker node (which mounts shared_dir rw) executing subhkl cleanup-shared!
                    os.chmod(shared_dir, 0o777)
                    try:
                        cli_res = CliRunner().invoke(
                            cli_app,
                            ["cleanup-shared", str(tool_inputs["networkPath"])],
                        )
                        assert cli_res.exit_code == 0
                    finally:
                        os.chmod(shared_dir, 0o555)
                    return {
                        "jobs": [{"id": "j_cleanup_1"}],
                        "outputs": [
                            {
                                "id": "ds_cleanup_log_1",
                                "name": "cleanup.log",
                                "file_ext": "txt",
                            }
                        ],
                    }
                raise AssertionError(f"Unexpected tool {tool_id}")

        class _FakeDatasets:
            def show_dataset(self, ds_id: str):
                return {"id": ds_id, "name": ds_id, "state": "ok", "deleted": False}

            def get_datasets(self, **_kw: object):
                return []

            def wait_for_dataset(self, *_a: object, **_kw: object):
                return None

        fake_store = SimpleNamespace(
            history_id="hist_ro",
            nova_connection=SimpleNamespace(
                galaxy_instance=SimpleNamespace(
                    base_url="http://localhost:8080",
                    tools=_FakeTools(),
                    datasets=_FakeDatasets(),
                )
            ),
        )
        monkeypatch.setattr(runner, "_api_request", _fake_api)
        monkeypatch.setattr(runner, "_download_dataset_to_file", _fake_dl)
        monkeypatch.setattr(runner, "get_datastore", lambda *_a, **_k: fake_store)
        monkeypatch.setattr(runner, "_has_tool_on_galaxy", lambda _t: True)
        monkeypatch.setattr(runner, "_get_tool_version", lambda _t: "0.3.0")
        monkeypatch.setattr(
            runner, "attach_dataset_cell_metadata_from_file", lambda *_a, **_k: None
        )

        # 3. UI receives pointer on ro mount, symlinks merged.h5, and reads detector frames cleanly
        rec_red: dict[str, object] = {
            "job_id": "r_ro",
            "stage_id": "reduce",
            "stdout": "",
        }
        runner._dispatch_nova_galaxy_thread(
            "reduce",
            {
                "instrument": "CG4D",
                "nexus_files": ["/HFIR/CG4D/IPTS-999999/nexus/CG4D_999999.nxs.h5"],
                "cell": [50, 60, 70, 90, 90, 90],
                "space_group": "P 21 21 21",
            },
            rec_red,
        )
        assert not rec_red.get("stderr"), f"Unexpected reduce error: {rec_red}"
        out_merged = str(rec_red["output_h5_path"])
        assert os.path.islink(out_merged)
        with h5py.File(out_merged, "r") as ro_hf:
            assert ro_hf["images"].shape == (1, 4, 4)

        # 4. On dataset switch / shutdown, local unlink is denied on ro mount -> UI dispatches worker cleanup tool!
        removed = runner.release_shared_scan_intermediates(
            keep_work_dir=str(nvme_dir / "subhkl_work_NEXT"),
            keep_paths=set(),
            wait_remote=True,
        )
        assert os.path.abspath(shared_merged_local) in removed
        assert len(dispatched_cleanup) == 1
        assert dispatched_cleanup[0][0] == "imaginex_subhkl_cleanup_shared"
        assert dispatched_cleanup[0][1]["networkPath"] == shared_merged_net
        assert not os.path.exists(shared_merged_local)
        assert "ds_scan_ro_1" in purged_ids
        assert "ds_cleanup_log_1" in purged_ids
    finally:
        os.chmod(shared_dir, 0o777)


def test_arcade_exit_quips_and_banner_html() -> None:
    """Verify all 230 cheerful space-group exit quips and the pixelated GAME OVER exit modal."""
    from subhkl.viz.arcade_exit_quips import (
        ARCADE_EXIT_QUIPS,
        ARCADE_PIXEL_FONT_DATA_URI,
        build_arcade_exit_quips_json,
    )
    from subhkl.viz.rainbow_zones import render_rainbow_html

    assert len(ARCADE_EXIT_QUIPS) == 230
    assert len(set(ARCADE_EXIT_QUIPS)) == 230
    assert all(isinstance(q, str) and len(q.strip()) >= 20 for q in ARCADE_EXIT_QUIPS)
    assert ARCADE_PIXEL_FONT_DATA_URI.startswith("data:font/ttf;base64,")
    parsed = json.loads(build_arcade_exit_quips_json())
    assert parsed == list(ARCADE_EXIT_QUIPS)

    sample_payload = {
        "instrument": "CG4D",
        "space_group": "P 21 21 21",
        "peaks": [],
        "zones": [],
        "panels": [],
    }
    html_arcade = render_rainbow_html(sample_payload, default_mode="arcade")
    assert 'id="btn-exit-app"' in html_arcade
    assert ">\n        ⏻\n      </button>" in html_arcade
    assert "⏻ Exit" not in html_arcade
    assert 'id="modal-arcade-exit"' in html_arcade
    assert 'font-family: "SubhklArcade"' in html_arcade
    assert ARCADE_PIXEL_FONT_DATA_URI in html_arcade
    assert 'id="btn-arcade-exit-cancel"' in html_arcade
    assert "Go back" in html_arcade
    assert "Another vibe" not in html_arcade
    assert (
        'id="btn-arcade-exit-close-tab" type="button" class="arcade-exit-btn" disabled'
        in html_arcade
    )
    assert "#btn-arcade-exit-close-tab {\n    border-color: #ef4444;" in html_arcade
    assert "arcade-exit-pixel-svg" not in html_arcade
    assert "AudioContext" not in html_arcade
    assert "🎲" not in html_arcade
    assert "✋" not in html_arcade
    assert "window.ARCADE_EXIT_QUIPS = " in html_arcade
    assert 'phase: "prepare"' in html_arcade
    assert 'phase: "confirm"' in html_arcade
    assert "restorePreviousStepAfterCancel" in html_arcade

    html_pro = render_rainbow_html(sample_payload, default_mode="pro")
    assert 'id="btn-exit-app"' in html_pro
    assert 'const UI_MODE = "pro".toLowerCase();' in html_pro


def test_exit_endpoint_cleans_shared_and_staging_and_stops_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify two-phase POST /api/pipeline/exit defers cleanup in Local prepare and sweeps + stops on confirm."""
    import socket
    import threading
    import time
    import urllib.request

    from subhkl.viz.nova_rainbow_view import (
        register_temp_artifact,
        serve_rainbow_zones,
    )

    nvme_dir = tmp_path / "exit_nvme"
    shared_dir = tmp_path / "exit_shared"
    nvme_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_dir))
    monkeypatch.setenv("SUBHKL_SHARED_DIR", str(shared_dir))
    monkeypatch.delenv("SUBHKL_EXEC_MODE", raising=False)
    monkeypatch.delenv("EP_PATH", raising=False)
    monkeypatch.delenv("GALAXY_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    staging_tmp = nvme_dir / "temp_stage_buf.h5"
    staging_tmp.write_bytes(b"STAGE_TMP")
    register_temp_artifact(str(staging_tmp))

    shared_merged, _ = allocate_shared_artifact_path("merged.h5", tag="EXIT_HTTP")
    with h5py.File(shared_merged, "w") as mf:
        mf.attrs["instrument"] = "CG4D"
        mf.create_dataset("images", data=np.zeros((1, 2, 2), dtype=np.float32))
    register_temp_artifact(shared_merged)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])

    srv_err: list[BaseException] = []

    def _run_srv() -> None:
        try:
            serve_rainbow_zones(
                payload={
                    "instrument": "CG4D",
                    "space_group": "P 1",
                    "peaks": [],
                    "zones": [],
                    "panels": [],
                },
                host="127.0.0.1",
                port=port,
                prefer_trame=False,
            )
        except (OSError, RuntimeError) as exc:  # pragma: no cover
            srv_err.append(exc)

    t = threading.Thread(target=_run_srv, daemon=True)
    t.start()

    # Wait until server responds on /api/pipeline/status
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(40):
        try:
            with urllib.request.urlopen(
                f"{base_url}/api/pipeline/status", timeout=1.0
            ) as r:
                if r.status == 200:
                    break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("serve_rainbow_zones did not become ready in time")

    # Phase 1 ("prepare") in Local mode: defers cleanup so "Go back" keeps local state intact, and keeps server alive
    req_prep = urllib.request.Request(
        f"{base_url}/api/pipeline/exit",
        data=json.dumps({"phase": "prepare"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_prep, timeout=5.0) as resp_prep:
        assert resp_prep.status == 200
        doc_prep = json.loads(resp_prep.read().decode("utf-8"))

    assert doc_prep["ok"] is True
    assert doc_prep["status"] == "prepared"
    assert doc_prep["deferred_cleanup"] is True
    assert doc_prep["restore_plan"]["mode"] == "local_deferred"
    assert staging_tmp.exists()
    assert os.path.exists(shared_merged)

    # Server remains responsive if the user clicks "Go back"
    with urllib.request.urlopen(
        f"{base_url}/api/pipeline/status", timeout=1.0
    ) as r_alive:
        assert r_alive.status == 200

    # Phase 2 ("confirm", e.g. text/plain sendBeacon on "Close tab"): sweeps files and stops server cleanly
    req = urllib.request.Request(
        f"{base_url}/api/pipeline/exit",
        data=json.dumps({"phase": "confirm"}).encode("utf-8"),
        headers={"Content-Type": "text/plain;charset=UTF-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        doc = json.loads(resp.read().decode("utf-8"))

    assert doc["ok"] is True
    assert doc["status"] == "exited"
    assert doc["deferred_cleanup"] is False
    assert doc["cleaned_staging_count"] >= 2
    assert not staging_tmp.exists()
    assert not os.path.exists(shared_merged)

    t.join(timeout=3.0)
    assert not t.is_alive(), (
        "Server thread should stop cleanly after /api/pipeline/exit confirm"
    )
    assert srv_err == []


def test_galaxy_exit_prepare_cleans_immediately_and_builds_restore_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In Galaxy mode, prepare cleans intermediates immediately and returns a Galaxy History restore plan."""
    from subhkl.viz.nova_rainbow_view import (
        NovaGalaxyPipelineRunner,
        register_temp_artifact,
    )

    nvme_dir = tmp_path / "gx_exit_nvme"
    nvme_dir.mkdir()
    monkeypatch.setenv("SUBHKL_STAGING_DIR", str(nvme_dir))

    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://galaxy.example.org",
        galaxy_api_key="fake-api-key",
        history_id="hist_123",
    )

    staged_solve = nvme_dir / "galaxy_ds_solve_42_solve.h5"
    staged_solve.write_bytes(b"SOLVE_TMP")
    register_temp_artifact(str(staged_solve))
    runner._dataset_id_by_path[os.path.abspath(str(staged_solve))] = "ds_solve_42"

    monkeypatch.setattr(
        runner,
        "list_history_datasets",
        lambda limit=200: [
            {"id": "ds_solve_42", "name": "CG4D_solve.h5", "state": "ok", "hid": 10},
            {
                "id": "ds_mtz_9",
                "name": "careless_merged.mtz",
                "state": "ok",
                "hid": 11,
            },
        ],
    )
    swept_calls: list[bool] = []
    monkeypatch.setattr(
        runner,
        "release_shared_scan_intermediates",
        lambda wait_remote=False: (
            swept_calls.append(bool(wait_remote)) or ["/shared/merged.h5"]
        ),
    )

    res = runner.execute_app_exit(
        phase="prepare",
        current_h5=str(staged_solve),
        payload={"artifact_kind": "solve", "frame_index": 3, "instrument": "CG4D"},
        mtz_overlay_info={"id": "ds_mtz_9"},
    )
    assert res["ok"] is True
    assert res["status"] == "prepared"
    assert res["deferred_cleanup"] is False
    assert swept_calls == [True]
    assert not staged_solve.exists()
    assert res["restore_plan"] == {
        "mode": "galaxy_dataset",
        "dataset_id": "ds_solve_42",
        "mtz_dataset_id": "ds_mtz_9",
        "frame_index": 3,
        "artifact_kind": "solve",
    }


def test_careless_merge_reports_runner_host_local_and_galaxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Careless ('merge') surfaces runner_host in local container, Galaxy polling, and UI."""
    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import render_rainbow_html

    html = render_rainbow_html(
        {
            "instrument": "CG4D",
            "space_group": "P 1",
            "peaks": [],
            "zones": [],
            "panels": [],
        }
    )
    assert "function escTxt(s)" in html
    assert "syncRunningHostDisplay" in html
    assert "[4. MERGE • ${hostTag}]" in html
    assert "Careless Variational Merge</strong> on host" in html

    # 1. Galaxy polling for stage_id == "merge" extracts runner_host from stdout or uname metric and preserves base_log
    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://galaxy.example.org",
        galaxy_api_key="fake-api-key",
        history_id="hist_merge",
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)

    calls = {"n": 0}

    def _fake_api(
        method: str, path: str, payload: object = None, **kwargs: object
    ) -> dict[str, object]:
        _ = (method, payload, kwargs)
        if "/console_output" in path:
            return {
                "stdout": "[pulsar] Runner host: gpu-worker-07\nIteration 1000/10000\n",
                "stderr": "",
            }
        calls["n"] += 1
        if calls["n"] == 1:
            return {"state": "running", "command_line": "careless poly ..."}
        return {"state": "deleted", "command_line": "careless poly ..."}

    monkeypatch.setattr(runner, "_api_request", _fake_api)
    job_rec: dict[str, object] = {
        "job_id": "gx_merge_99",
        "stage_id": "merge",
        "mode": "galaxy",
        "state": "queued",
        "stdout": "[galaxy -> pulsar] Dispatched imaginex_subhkl_merge\n",
        "stderr": "",
    }
    runner._poll_galaxy_job(job_rec, final_step=False)
    assert job_rec["runner_host"] == "gpu-worker-07"
    assert "[galaxy -> pulsar] Dispatched imaginex_subhkl_merge" in str(
        job_rec["stdout"]
    )
    assert "[pulsar] Runner host: gpu-worker-07" in str(job_rec["stdout"])


def test_galaxy_careless_merge_completes_without_h5_signature_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: NDIP ``imaginex_subhkl_merge`` must not feed ``metrics_json`` to ``h5py.File``."""
    import gemmi
    import h5py
    import numpy as np

    from subhkl.viz.nova_rainbow_view import NovaGalaxyPipelineRunner
    from subhkl.viz.rainbow_zones import (
        build_rainbow_zone_payload,
        infer_dataset_extension,
        render_rainbow_html,
    )

    assert infer_dataset_extension("merge_table.csv", None) == "csv"
    assert infer_dataset_extension("merge_table_csv", None) == "csv"
    assert infer_dataset_extension("metrics_json", None) == "json"
    assert infer_dataset_extension("merged_mtz", None) == "mtz"

    html = render_rainbow_html(
        {
            "instrument": "CG4D",
            "space_group": "I a -3 d",
            "peaks": [],
            "zones": [],
            "panels": [],
        }
    )
    assert 'stageId !== "merge" && Array.isArray(aj.outputs)' in html

    # Build a read-only shared integrated.h5 and symlink it into work_dir/integrated.h5
    shared_ro_dir = tmp_path / "shared_ro"
    shared_ro_dir.mkdir()
    shared_int_h5 = shared_ro_dir / "integrated_shared.h5"
    with h5py.File(shared_int_h5, "w") as hf:
        hf["instrument"] = b"CG4D"
        hf["wavelengths"] = np.array([2.0, 4.0], dtype=np.float64)
        sgrp = hf.create_group("sample")
        sgrp["space_group"] = b"I a -3 d"
        sgrp["cell"] = np.array(
            [11.93, 11.93, 11.93, 90.0, 90.0, 90.0], dtype=np.float64
        )
        sgrp["B"] = np.diag([1.0 / 11.93, 1.0 / 11.93, 1.0 / 11.93])
        sgrp["U"] = np.eye(3, dtype=np.float64)
        ggrp = hf.create_group("goniometer")
        ggrp["R"] = np.eye(3, dtype=np.float64)[None, :, :]
        pgrp = hf.create_group("peaks")
        pgrp["xyz"] = np.array([[0.25, 0.0, -1.0], [0.0, 0.25, -1.0]], dtype=np.float64)
        pgrp["h"] = np.array([2, 0], dtype=np.int32)
        pgrp["k"] = np.array([0, 2], dtype=np.int32)
        pgrp["l"] = np.array([0, 0], dtype=np.int32)
        pgrp["intensity"] = np.array([120.0, 95.0], dtype=np.float64)
        pgrp["sigma"] = np.array([6.0, 5.0], dtype=np.float64)
        pgrp["run_index"] = np.array([0, 0], dtype=np.int32)
        pgrp["wavelength"] = np.array([2.4, 2.8], dtype=np.float64)
        pgrp["two_theta"] = np.array([0.35, 0.35], dtype=np.float64)
        pgrp["azimuthal"] = np.array([0.0, 1.57], dtype=np.float64)
        pgrp["var_u"] = np.array([2.25, 2.25], dtype=np.float64)
        pgrp["var_v"] = np.array([2.25, 2.25], dtype=np.float64)
    shared_int_h5.chmod(0o444)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    work_int_h5 = work_dir / "integrated.h5"
    work_int_h5.symlink_to(shared_int_h5)

    # Create valid minimal MTZ binaries via gemmi for merged.mtz, predictions.mtz, xval.mtz
    mtz_build_dir = tmp_path / "mtz_src"
    mtz_build_dir.mkdir()

    def _make_merged_mtz(dest: Path) -> bytes:
        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi.find_spacegroup_by_name("I a -3 d")
        mtz.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90, 90, 90))
        mtz.add_dataset("crystal")
        mtz.add_column("I", "J")
        mtz.add_column("SigI", "Q")
        mtz.add_column("N", "I")
        mtz.set_data(
            np.array(
                [
                    [2.0, 0.0, 0.0, 118.0, 5.5, 2.0],
                    [0.0, 2.0, 0.0, 96.0, 5.0, 2.0],
                ],
                dtype=np.float32,
            )
        )
        mtz.write_to_file(str(dest))
        return dest.read_bytes()

    def _make_pred_mtz(dest: Path) -> bytes:
        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi.find_spacegroup_by_name("I a -3 d")
        mtz.set_cell_for_all(gemmi.UnitCell(11.93, 11.93, 11.93, 90, 90, 90))
        mtz.add_dataset("crystal")
        for lbl, ctype in (
            ("Batch", "B"),
            ("Iobs", "J"),
            ("SigIobs", "Q"),
            ("Ipred", "J"),
            ("SigIpred", "Q"),
            ("Scale", "R"),
        ):
            mtz.add_column(lbl, ctype)
        mtz.set_data(
            np.array(
                [
                    [2.0, 0.0, 0.0, 1.0, 120.0, 6.0, 117.5, 5.2, 1.02],
                    [0.0, 2.0, 0.0, 1.0, 95.0, 5.0, 96.2, 4.8, 0.99],
                ],
                dtype=np.float32,
            )
        )
        mtz.write_to_file(str(dest))
        return dest.read_bytes()

    merged_bytes = _make_merged_mtz(mtz_build_dir / "merged.mtz")
    pred_bytes = _make_pred_mtz(mtz_build_dir / "predictions.mtz")
    xval_bytes = merged_bytes
    metrics_payload = {
        "cc_half": 0.964,
        "cc_pred": 0.981,
        "r_pim": 0.042,
        "r_meas": 0.061,
        "i_over_sigma": 18.5,
        "completeness": 0.945,
        "multiplicity": 4.2,
        "n_unique": 2,
        "n_observations": 2,
    }
    metrics_bytes = json.dumps(metrics_payload).encode("utf-8")
    csv_bytes = b"shell,d_max,d_min,cc_half\nall,10.0,1.0,0.964\n"
    html_bytes = b"<html><body><table><tr><td>CC1/2</td><td>0.964</td></tr></table></body></html>"

    dataset_blobs = {
        "ds_merged": merged_bytes,
        "ds_pred": pred_bytes,
        "ds_xval": xval_bytes,
        "ds_metrics": metrics_bytes,
        "ds_csv": csv_bytes,
        "ds_html": html_bytes,
    }

    runner = NovaGalaxyPipelineRunner(
        galaxy_url="https://galaxy.example.org",
        galaxy_api_key="fake-api-key",
        history_id="6644e5c0e8a790e1",
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(runner, "_upload_compact_ndip_artifact", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "get_datastore", lambda *_a, **_k: object())

    # Simulate Galaxy's run_tool response where file_ext can be empty or 'json'
    raw_run_tool_outputs = [
        {
            "id": "ds_merged",
            "output_name": "merged_mtz",
            "name": "merged.mtz",
            "file_ext": "",
        },
        {
            "id": "ds_pred",
            "output_name": "predictions_mtz",
            "name": "predictions.mtz",
            "file_ext": "mtz",
        },
        {
            "id": "ds_xval",
            "output_name": "xval_mtz",
            "name": "xval.mtz",
            "file_ext": "mtz",
        },
        {
            "id": "ds_metrics",
            "output_name": "metrics_json",
            "name": "careless_metrics.json",
            "file_ext": "json",
        },
        {
            "id": "ds_csv",
            "output_name": "merge_table_csv",
            "name": "merge_table.csv",
            "file_ext": "",
        },
        {
            "id": "ds_html",
            "output_name": "merge_table_html",
            "name": "merge_table.html",
            "file_ext": "html",
        },
    ]
    norm_outputs = runner._normalize_galaxy_job_outputs(
        raw_run_tool_outputs, stage_id="merge"
    )
    assert [o["extension"] for o in norm_outputs] == [
        "mtz",
        "mtz",
        "mtz",
        "json",
        "csv",
        "html",
    ]

    def _fake_api(
        method: str, path: str, payload: object = None, **kwargs: object
    ) -> object:
        _ = (method, payload)
        if kwargs.get("raw"):
            for dsid, blob in dataset_blobs.items():
                if f"/api/datasets/{dsid}/display" in path:
                    return blob
            raise AssertionError(f"Unexpected raw dataset request: {path}")
        if "/console_output" in path:
            return {
                "stdout": "[pulsar] Runner host: IMAGINEX-2\nCareless version 0.5.4\n",
                "stderr": "Merging repeat 1 half 2: 100%\n",
            }
        if path.startswith("/api/jobs/"):
            return {
                "state": "ok",
                "command_line": "careless poly ...",
                "outputs": {
                    "merged_mtz": {"id": "ds_merged", "src": "hda"},
                    "predictions_mtz": {"id": "ds_pred", "src": "hda"},
                    "xval_mtz": {"id": "ds_xval", "src": "hda"},
                    "metrics_json": {"id": "ds_metrics", "src": "hda"},
                    "merge_table_csv": {"id": "ds_csv", "src": "hda"},
                    "merge_table_html": {"id": "ds_html", "src": "hda"},
                },
            }
        return {}

    monkeypatch.setattr(runner, "_api_request", _fake_api)

    completed_paths: list[str] = []
    reloaded_payloads: list[dict[str, object]] = []

    def _on_complete(h5_path: str) -> None:
        completed_paths.append(h5_path)
        reloaded_payloads.append(
            build_rainbow_zone_payload(replay_h5=h5_path, frame_index=0)
        )

    job_rec: dict[str, object] = {
        "job_id": "5af8bc9400aa8f53",
        "stage_id": "merge",
        "mode": "galaxy",
        "state": "running",
        "stdout": "[galaxy -> pulsar] Dispatched imaginex_subhkl_merge (Galaxy job_id=5af8bc9400aa8f53) -> Pulsar GPU queue\n",
        "stderr": "",
        "outputs": norm_outputs,
        "params": {
            "instrument": "CG4D",
            "space_group": "I a -3 d",
            "cell": [11.93, 11.93, 11.93, 90.0, 90.0, 90.0],
            "wavelength_range": [2.0, 4.0],
            "d_min": 1.0,
            "peaks_file": str(work_int_h5),
        },
    }

    runner._poll_galaxy_job(
        job_rec,
        on_complete_h5=_on_complete,
        final_step=True,
        work_dir=str(work_dir),
    )

    stdout_txt = str(job_rec["stdout"])
    assert "Warning fetching job output" not in stdout_txt
    assert "file signature not found" not in stdout_txt
    assert job_rec["state"] == "ok"
    assert job_rec["runner_host"] == "IMAGINEX-2"
    assert job_rec.get("updated_payload_ready") is True
    assert job_rec.get("output_h5_path") == str(work_int_h5)
    assert completed_paths == [str(work_int_h5)]

    # All 6 Careless artifacts must be downloaded into work_dir
    for fname in (
        "merged.mtz",
        "predictions.mtz",
        "xval.mtz",
        "careless_metrics.json",
        "merge_table.csv",
        "merge_table.html",
    ):
        assert (work_dir / fname).is_file(), f"Missing {fname} in {work_dir}"

    # Read-only shared target must remain untouched while work_dir/integrated.h5 is materialized & stamped
    assert not work_int_h5.is_symlink()
    with h5py.File(shared_int_h5, "r") as hf_ro:
        assert "careless" not in hf_ro
    with h5py.File(work_int_h5, "r") as hf_stamped:
        assert "careless/metrics_json" in hf_stamped

    assert len(reloaded_payloads) == 1
    payload = reloaded_payloads[0]
    assert payload["artifact_kind"] == "merged_careless"
    assert isinstance(payload.get("merge_metrics"), dict)
    assert payload["merge_metrics"]["cc_half"] == pytest.approx(0.964)
    assert payload["merge_metrics"]["r_pim"] == pytest.approx(0.042)
