"""Tests for Expectation-Maximization per-pixel (valid, dark, bright) classifier."""

import copy
from types import SimpleNamespace

import h5py
import numpy as np

from subhkl.commands import run_peak_predictor
from subhkl.config import beamlines
from subhkl.instrument.detector import Detector
from subhkl.instrument.goniometer import calc_goniometer_rotation_matrix
from subhkl.scan_refine import run_scan_refinement
from subhkl.search.pixel_em import PixelEMClassifier


def _build_B_orthorhombic(a: float, b: float, c: float) -> np.ndarray:
    return np.diag([1.0 / a, 1.0 / b, 1.0 / c])


def _project_crystal_spots(
    counts_run: dict[int, np.ndarray],
    U: np.ndarray,
    B: np.ndarray,
    R_gon: np.ndarray,
    banks: list[int],
    wavelength: tuple[float, float],
    d_min: float,
    peak_amp: float = 28.0,
    sigma_px: float = 1.6,
) -> list[tuple[int, int, int]]:
    """Render Gaussian Bragg reflections onto active panels and return spot centers."""
    max_idx = int(np.ceil(max(1.0 / B[0, 0], 1.0 / B[1, 1], 1.0 / B[2, 2]) / d_min)) + 1
    grid = np.arange(-max_idx, max_idx + 1, dtype=np.float64)
    hh, kk, ll = np.meshgrid(grid, grid, grid, indexing="ij")
    hkl = np.column_stack([hh.ravel(), kk.ravel(), ll.ravel()])
    G0 = hkl @ (U @ B).T
    gn = np.linalg.norm(G0, axis=1)
    hkl_mask = (gn > 1e-6) & (gn <= 1.0 / d_min)
    G0 = G0[hkl_mask]

    G = G0 @ R_gon.T
    g2 = np.sum(G * G, axis=1)
    valid = (G[:, 2] < -1e-6) & (g2 > 1e-10)
    lam = np.zeros_like(g2)
    lam[valid] = -2.0 * G[valid, 2] / g2[valid]
    in_wl = valid & (lam >= wavelength[0]) & (lam <= wavelength[1])
    if not np.any(in_wl):
        return []

    G_sel = G[in_wl]
    lam_sel = lam[in_wl]
    kf = G_sel + np.array([0.0, 0.0, 1.0]) / lam_sel[:, None]
    kf = kf / np.linalg.norm(kf, axis=1, keepdims=True)

    landed: list[tuple[int, int, int]] = []
    rad = 4
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    kern = peak_amp * np.exp(-0.5 * (yy * yy + xx * xx) / (sigma_px * sigma_px))

    for bk in banks:
        det = Detector(dict(beamlines["CG4D"][str(bk)]))
        nrm = np.cross(det.uhat, det.vhat)
        nrm = nrm / np.linalg.norm(nrm)
        denom = kf @ nrm
        t_num = float(det.center @ nrm)
        hit = np.abs(denom) > 1e-6
        t_ray = np.where(hit, t_num / np.where(hit, denom, 1.0), -1.0)
        hit &= t_ray > 0.0
        if not np.any(hit):
            continue
        pts = t_ray[hit, None] * kf[hit] - det.center[None, :]
        u_c = (pts @ det.uhat) / det.width
        v_c = (pts @ det.vhat) / det.height
        in_pan = (u_c >= 0.05) & (u_c <= 0.95) & (v_c >= 0.05) & (v_c <= 0.95)
        if not np.any(in_pan):
            continue
        cols = np.round(u_c[in_pan] * (det.m - 1)).astype(int)
        rows = np.round(v_c[in_pan] * (det.n - 1)).astype(int)
        img = counts_run[bk]
        for r0, c0 in zip(rows, cols):
            r_lo, r_hi = max(0, r0 - rad), min(det.n, r0 + rad + 1)
            c_lo, c_hi = max(0, c0 - rad), min(det.m, c0 + rad + 1)
            img[r_lo:r_hi, c_lo:c_hi] += kern[
                (r_lo - r0 + rad) : (r_hi - r0 + rad),
                (c_lo - c0 + rad) : (c_hi - c0 + rad),
            ]
            landed.append((bk, int(r0), int(c0)))
    return landed


def _write_synthetic_dataset(tmp_path):
    """Create a 3-run, 3-bank HDF5 dataset with dead panel, shadow, and bright spots."""
    rng = np.random.default_rng(1234)
    banks = [61, 62, 63]
    cell = [28.0, 32.0, 36.0, 90.0, 90.0, 90.0]
    B = _build_B_orthorhombic(*cell[:3])
    U_true = np.eye(3)
    wl = (2.8, 4.2)
    d_min = 2.0
    axes = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    omegas = [0.0, 18.0, 36.0]

    # Off-lattice static bright locations on bank 62 (far from run-0 reflections)
    strong_bright_rc = (92, 412)
    moderate_bright_rc = (404, 108)

    all_imgs = []
    all_bank_ids = []
    all_angles = []
    offsets = []
    run0_spots: list[tuple[int, int, int]] = []

    yy, xx = np.mgrid[-3:4, -3:4]
    gauss_spot = np.exp(-0.5 * (yy * yy + xx * xx) / (1.5 * 1.5))

    for run_i, om in enumerate(omegas):
        offsets.append(len(all_imgs))
        R_gon = calc_goniometer_rotation_matrix(axes, np.array([om]))
        counts_run = {
            bk: rng.poisson(4.0, size=(512, 512)).astype(np.float64) for bk in banks
        }
        spots = _project_crystal_spots(
            counts_run, U_true, B, R_gon, banks, wl, d_min, peak_amp=26.0
        )
        if run_i == 0:
            run0_spots = spots

        # Static artifacts:
        # 1) Bank 63 is completely dead
        counts_run[63][:] = 0.0
        # 2) Bank 61 has a static cryostat shadow slab (rows 128:256, cols 128:256)
        counts_run[61][128:256, 128:256] = 0.0
        # 3) Bank 62 has a strong static Al-enclosure flare + a weaker static hot spot
        r_s, c_s = strong_bright_rc
        counts_run[62][r_s - 3 : r_s + 4, c_s - 3 : c_s + 4] += 260.0 * gauss_spot
        r_m, c_m = moderate_bright_rc
        counts_run[62][r_m - 3 : r_m + 4, c_m - 3 : c_m + 4] += 22.0 * gauss_spot

        for bk in banks:
            all_imgs.append(counts_run[bk])
            all_bank_ids.append(bk)
            all_angles.append([om])

    frames_path = tmp_path / "em_synth_frames.h5"
    with h5py.File(frames_path, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["images"] = np.asarray(all_imgs, dtype=np.float32)
        f["bank_ids"] = np.asarray(all_bank_ids, dtype=np.int32)
        f["file_offsets"] = np.asarray(offsets, dtype=np.int64)
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = np.asarray(all_angles, dtype=np.float64)
        f["goniometer/names"] = [b"omega"]
        f["instrument/wavelength"] = np.asarray(wl, dtype=np.float64)
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k}"] = float(v)
        f["sample/space_group"] = "P 2 2 2"

    boot_path = tmp_path / "em_seed.h5"
    with h5py.File(boot_path, "w") as f:
        f["sample/U"] = U_true
        f["sample/B"] = B

    return (
        frames_path,
        boot_path,
        U_true,
        B,
        cell,
        wl,
        d_min,
        strong_bright_rc,
        moderate_bright_rc,
        run0_spots,
    )


def test_pixel_em_single_and_multiframe_dark_bright_classification(tmp_path):
    (
        frames_path,
        _boot_path,
        U_true,
        B,
        cell,
        wl,
        d_min,
        strong_rc,
        mod_rc,
        run0_spots,
    ) = _write_synthetic_dataset(tmp_path)

    binning = 8
    br_s, bc_s = strong_rc[0] // binning, strong_rc[1] // binning
    br_m, bc_m = mod_rc[0] // binning, mod_rc[1] // binning

    # Single-frame (N=1) EM classification
    args_n1 = SimpleNamespace(
        nexus=str(frames_path),
        instrument="CG4D",
        runs=[0],
        stable_runs=[0],
        binning=binning,
        border_width=4,
        static_mask_file=None,
        wl_min=wl[0],
        wl_max=wl[1],
        assign_wavelength=wl,
        d_min=d_min,
        cell=cell,
        space_group="P 2 2 2",
    )
    em_n1 = PixelEMClassifier.from_args(args_n1)
    assert em_n1 is not None
    em_n1.update_e_step(U_true, B)

    # Dead bank 63 and shadowed slab on bank 61 are dark even from N=1
    assert float(np.mean(em_n1.gamma_dark[63])) > 0.95
    assert float(np.mean(em_n1.gamma_dark[61][18:30, 18:30])) > 0.90
    assert float(np.mean(em_n1.gamma_valid[61][4:12, 4:12])) > 0.90

    # Strong off-lattice Al flare is detected on N=1
    gb_strong_n1 = float(
        np.max(em_n1.gamma_bright[62][br_s - 1 : br_s + 2, bc_s - 1 : bc_s + 2])
    )
    gb_mod_n1 = float(
        np.max(em_n1.gamma_bright[62][br_m - 1 : br_m + 2, bc_m - 1 : bc_m + 2])
    )
    assert gb_strong_n1 > 0.75

    # Genuine Bragg reflections on bank 62 remain valid (not misclassified bright)
    b62_spots = [
        (r, c)
        for (bk, r, c) in run0_spots
        if bk == 62
        and (abs(r - strong_rc[0]) > 24 or abs(c - strong_rc[1]) > 24)
        and (abs(r - mod_rc[0]) > 24 or abs(c - mod_rc[1]) > 24)
        and 16 <= r < 496
        and 16 <= c < 496
    ]
    assert len(b62_spots) >= 3
    for r_sp, c_sp in b62_spots[:8]:
        assert em_n1.gamma_valid[62][r_sp // binning, c_sp // binning] > 0.80

    # Candidate reweighting strongly prefers U_true over a wrong orientation
    ev_true = em_n1.evaluate_candidate(U_true, B)
    c_ang, s_ang = np.cos(np.deg2rad(27.0)), np.sin(np.deg2rad(27.0))
    U_wrong = np.array([[c_ang, -s_ang, 0.0], [s_ang, c_ang, 0.0], [0.0, 0.0, 1.0]])
    ev_wrong = em_n1.evaluate_candidate(U_wrong, B)
    assert ev_true["em_score"] > ev_wrong["em_score"] + 5.0

    # Multi-frame (N=3) EM classification: weaker static spot becomes bright
    # thanks to persistence across goniometer frames, while transient Bragg
    # reflections stay valid.
    args_n3 = SimpleNamespace(**{**vars(args_n1), "runs": [0, 1, 2]})
    em_n3 = PixelEMClassifier.from_args(args_n3)
    assert em_n3 is not None
    em_n3.update_e_step(U_true, B)

    gb_mod_n3 = float(
        np.max(em_n3.gamma_bright[62][br_m - 1 : br_m + 2, bc_m - 1 : bc_m + 2])
    )
    assert gb_mod_n3 > gb_mod_n1
    assert gb_mod_n3 > 0.75
    for r_sp, c_sp in b62_spots[:8]:
        assert em_n3.gamma_valid[62][r_sp // binning, c_sp // binning] > 0.85


def test_scan_refinement_and_peak_predictor_mask_dark_and_bright_pixels(tmp_path):
    (
        frames_path,
        boot_path,
        _U_true,
        _B,
        cell,
        wl,
        d_min,
        strong_rc,
        _mod_rc,
        _run0_spots,
    ) = _write_synthetic_dataset(tmp_path)

    solve_out = tmp_path / "em_solve.h5"
    pred_out = tmp_path / "em_pred.h5"
    res = run_scan_refinement(
        frames_path,
        solve_out,
        bootstrap=boot_path,
        instrument="CG4D",
        cell=cell,
        space_group="P 2 2 2",
        runs=[0, 1, 2],
        stable_runs=[0, 1, 2],
        modes=[],
        wavelength=wl,
        assignment_wavelength=wl,
        d_min=d_min,
        n_zones=60,
        binning=8,
        maxiter=5,
    )
    assert res["status"] in ("converged", "not_converged")

    with h5py.File(solve_out, "r") as fs:
        assert "detector/pixel_mask" in fs
        assert "detector/pixel_class" in fs
        bids = list(np.asarray(fs["detector/mask_bank_ids"][()], dtype=int))
        pm = fs["detector/pixel_mask"][()]
        pc = fs["detector/pixel_class"][()]
        assert bids == [61, 62, 63]
        # Dead bank 63 is masked dark (class 1)
        assert np.all(pm[bids.index(63)] == 0)
        assert np.all(pc[bids.index(63)] == 1)
        # Shadow interior on bank 61 is masked dark
        assert np.all(pm[bids.index(61), 150:230, 150:230] == 0)
        # Strong static bright spot on bank 62 is masked bright (class 2)
        r_s, c_s = strong_rc
        assert np.any(pc[bids.index(62), r_s - 8 : r_s + 8, c_s - 8 : c_s + 8] == 2)

    saved_cg4d = copy.deepcopy(beamlines["CG4D"])
    try:
        run_peak_predictor(
            filename=str(frames_path),
            instrument="CG4D",
            indexed_hdf5_filename=str(solve_out),
            integration_peaks_filename=str(pred_out),
            d_min=d_min,
            wavel_min=wl[0],
            wavel_max=wl[1],
            max_workers=1,
        )
    finally:
        beamlines["CG4D"] = saved_cg4d

    with h5py.File(pred_out, "r") as fp:
        assert "detector/pixel_mask" in fp
        pm_out = fp["detector/pixel_mask"][()]
        m_bids = {int(b): i for i, b in enumerate(fp["detector/mask_bank_ids"][()])}
        p_bids = fp["bank_ids"][()]
        total_peaks = 0
        for img_k in fp["banks"]:
            phys_b = int(p_bids[int(img_k)])
            ii = np.clip(np.round(fp["banks"][img_k]["i"][()]).astype(int), 0, 511)
            jj = np.clip(np.round(fp["banks"][img_k]["j"][()]).astype(int), 0, 511)
            total_peaks += len(ii)
            assert phys_b != 63 or len(ii) == 0
            if len(ii) > 0:
                assert np.all(pm_out[m_bids[phys_b], ii, jj] == 1)
        assert total_peaks > 0


def test_overdispersed_diffuse_gradient_not_flagged_bright(tmp_path):
    """Smooth ~12% diffuse scattering curvature across 4 frames stays valid."""
    rng = np.random.default_rng(99)
    yy, xx = np.mgrid[0:512, 0:512]
    # Smooth high-background diffuse halo (~40 counts/px = 2560 counts per 8x8 bin)
    halo = 36.0 + 10.0 * np.exp(
        -(((yy - 256.0) / 90.0) ** 2 + ((xx - 256.0) / 90.0) ** 2)
    )
    axes = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    omegas = [0.0, 12.0, 24.0, 36.0]

    imgs = []
    for _ in omegas:
        im = rng.poisson(halo).astype(np.float32)
        # Inject one sharp 3x3 static hot defect at (100, 100)
        im[99:102, 99:102] += 450.0
        imgs.append(im)

    fpath = tmp_path / "diffuse_halo.h5"
    cell = [18.0, 21.0, 24.0, 90.0, 90.0, 90.0]
    with h5py.File(fpath, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["images"] = np.asarray(imgs, dtype=np.float32)
        f["bank_ids"] = np.asarray([62, 62, 62, 62], dtype=np.int32)
        f["file_offsets"] = np.asarray([0, 1, 2, 3], dtype=np.int64)
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = np.asarray([[o] for o in omegas], dtype=np.float64)
        f["goniometer/names"] = [b"omega"]
        f["instrument/wavelength"] = np.asarray([2.8, 4.2], dtype=np.float64)
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k}"] = float(v)
        f["sample/space_group"] = "P 2 2 2"

    args = SimpleNamespace(
        nexus=str(fpath),
        instrument="CG4D",
        runs=[0, 1, 2, 3],
        stable_runs=[0, 1, 2, 3],
        binning=8,
        border_width=4,
        static_mask_file=None,
        wl_min=2.8,
        wl_max=4.2,
        assign_wavelength=(2.8, 4.2),
        d_min=2.5,
        cell=cell,
        space_group="P 2 2 2",
    )
    em = PixelEMClassifier.from_args(args)
    assert em is not None
    em.update_e_step(np.eye(3), _build_B_orthorhombic(*cell[:3]))

    gb = em.gamma_bright[62]
    # Sharp hot spot near bin (12, 12) is flagged bright
    assert float(np.max(gb[11:14, 11:14])) > 0.85
    # Smooth diffuse halo center around (32, 32) has zero false bright bins
    assert int(np.sum(gb[20:44, 20:44] > 0.5)) == 0
