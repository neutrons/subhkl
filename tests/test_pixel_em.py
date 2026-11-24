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


def test_single_frame_coarse_dmin_protects_high_q_bragg_spots_and_masks_flare(tmp_path):
    """Single-frame coarse d_min=2.8 protects real d<2.8 A Bragg peaks while masking Al flares."""
    rng = np.random.default_rng(2523)
    banks = [12, 62]
    cell = [28.0, 32.0, 36.0, 90.0, 90.0, 90.0]
    B = _build_B_orthorhombic(*cell[:3])
    U_true = np.eye(3)
    wl = (2.8, 4.5)
    axes = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    R_gon = calc_goniometer_rotation_matrix(axes, [11.0])

    counts = {bk: np.full((512, 512), 14.0, dtype=np.float64) for bk in banks}
    spots_all = _project_crystal_spots(
        counts,
        U_true,
        B,
        R_gon,
        banks,
        wl,
        d_min=1.75,
        peak_amp=52.0,
        sigma_px=1.6,
    )
    spots_coarse = set(
        _project_crystal_spots(
            {bk: np.zeros((512, 512), dtype=np.float64) for bk in banks},
            U_true,
            B,
            R_gon,
            banks,
            wl,
            d_min=2.8,
        )
    )
    high_q_spots = [s for s in spots_all if s not in spots_coarse]
    assert len(high_q_spots) >= 5

    # Choose an off-lattice flare site on bank 62 at least 30 px from every crystal spot
    flare_rc = (88, 424)
    b62_spots = [(r, c) for (bk, r, c) in spots_all if bk == 62]
    if b62_spots:
        min_d = min(np.hypot(r - flare_rc[0], c - flare_rc[1]) for (r, c) in b62_spots)
        assert min_d > 28.0
    counts[62][
        flare_rc[0] - 2 : flare_rc[0] + 2, flare_rc[1] - 2 : flare_rc[1] + 2
    ] += 1400.0

    imgs = [rng.poisson(counts[bk]).astype(np.float32) for bk in banks]
    fpath = tmp_path / "single_frame_d28.h5"
    with h5py.File(fpath, "w") as f:
        f.attrs["instrument"] = "CG4D"
        f["images"] = np.asarray(imgs, dtype=np.float32)
        f["bank_ids"] = np.asarray(banks, dtype=np.int32)
        f["file_offsets"] = np.asarray([0], dtype=np.int64)
        f["goniometer/axes"] = axes
        f["goniometer/angles"] = np.asarray([[11.0], [11.0]], dtype=np.float64)
        f["goniometer/names"] = [b"omega"]
        f["instrument/wavelength"] = np.asarray(wl, dtype=np.float64)
        for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell):
            f[f"sample/{k}"] = float(v)
        f["sample/space_group"] = "P 2 2 2"

    args = SimpleNamespace(
        nexus=str(fpath),
        instrument="CG4D",
        runs=[0],
        stable_runs=[0],
        binning=8,
        border_width=4,
        static_mask_file=None,
        wl_min=wl[0],
        wl_max=wl[1],
        assign_wavelength=wl,
        d_min=2.8,
        cell=cell,
        space_group="P 2 2 2",
    )
    em = PixelEMClassifier.from_args(args)
    assert em is not None
    assert em._d_min_em < 2.0
    em.update_e_step(U_true, B)
    exported = em.export_pixel_masks()
    b_to_idx = {int(b): i for i, b in enumerate(exported.bank_ids)}
    p_cls = exported.pixel_class

    # Off-lattice Al flare is classified bright (class 2)
    assert np.any(
        p_cls[
            b_to_idx[62],
            flare_rc[0] - 6 : flare_rc[0] + 6,
            flare_rc[1] - 6 : flare_rc[1] + 6,
        ]
        == 2
    )
    # Every real crystal Bragg reflection (including d < 2.8 A high-Q spots) stays valid (class 0)
    for bk, r0, c0 in high_q_spots:
        assert p_cls[b_to_idx[bk], r0, c0] == 0


def test_jax_splat_and_2d_filters_match_cpu_scipy_reference(tmp_path):
    """Batched JAX median/75th-percentile filters and reflection splat match CPU references."""
    from scipy.ndimage import median_filter, percentile_filter

    from subhkl.search.pixel_em import (
        _batch_inpainted_median_bg,
        _run_batch_median_2d,
        _run_batch_pct75_2d,
        inpainted_median_bg,
    )

    rng = np.random.default_rng(1234)
    arr = rng.gamma(shape=3.5, scale=12.0, size=(5, 32, 32)).astype(np.float32)

    for k in (5, 7):
        got_med = _run_batch_median_2d(arr, k)
        ref_med = np.stack(
            [median_filter(a, size=k, mode="nearest") for a in arr], axis=0
        )
        np.testing.assert_allclose(got_med, ref_med, rtol=1e-5, atol=1e-5)

    got_p75 = _run_batch_pct75_2d(arr, 5)
    # Odd 5x5 window has 25 elements; rank index 18 matches scipy percentile=75
    ref_p75 = np.stack(
        [percentile_filter(a, percentile=75, size=5, mode="nearest") for a in arr],
        axis=0,
    )
    np.testing.assert_allclose(got_p75, ref_p75, rtol=1e-5, atol=1e-5)

    keys = [(idx // 2, 60 + (idx % 2)) for idx in range(arr.shape[0])]
    y_rb = {keys[idx]: arr[idx].astype(np.float64) for idx in range(arr.shape[0])}
    bg_raw_rb = {
        keys[idx]: median_filter(arr[idx], size=5, mode="nearest").astype(np.float64)
        for idx in range(arr.shape[0])
    }
    dark_masks = {}
    bright_masks = {}
    for idx, k_rb in enumerate(keys):
        dm = np.zeros(arr.shape[1:], dtype=bool)
        bm = np.zeros(arr.shape[1:], dtype=bool)
        if idx == 2:
            dm[:, :] = True  # all-dark bank edge case
        else:
            dm[8:16, 10:18] = True
            if idx % 2 == 0:
                bm[22:25, 22:25] = True
        dark_masks[k_rb] = dm
        bright_masks[k_rb] = bm

    got_bg = _batch_inpainted_median_bg(
        keys, y_rb, dark_masks, bright_masks, 5, bg_raw_rb
    )
    for k_rb in keys:
        ref_bg = inpainted_median_bg(
            y_rb[k_rb],
            dark_masks[k_rb],
            bright_masks[k_rb],
            5,
            bg_raw=bg_raw_rb[k_rb],
        )
        np.testing.assert_allclose(got_bg[k_rb], ref_bg, rtol=1e-5, atol=1e-5)

    (
        frames_path,
        _boot_path,
        U_true,
        B,
        cell,
        wl,
        d_min,
        _strong_rc,
        _mod_rc,
        _run0_spots,
    ) = _write_synthetic_dataset(tmp_path)
    args = SimpleNamespace(
        nexus=str(frames_path),
        instrument="CG4D",
        runs=[0, 1, 2],
        stable_runs=[0, 1, 2],
        binning=8,
        border_width=4,
        static_mask_file=None,
        wl_min=wl[0],
        wl_max=wl[1],
        assign_wavelength=wl,
        d_min=d_min,
        cell=cell,
        space_group="P 2 2 2",
    )
    em = PixelEMClassifier.from_args(args)
    assert em is not None

    kw = {
        "radial_scale": 0.035,
        "axial_scale": -0.02,
        "global_rot_vec": np.array([0.008, -0.012, 0.005], dtype=np.float64),
        "global_trans_m": np.array([0.001, -0.002, 0.0015], dtype=np.float64),
        "sample_shift_m": np.array([0.0005, -0.0003, 0.0004], dtype=np.float64),
    }

    jax_single = em._project_and_splat(U_true, B, match_sigma_px=5.0, **kw)
    cpu_single = em._project_and_splat_cpu(U_true, B, match_sigma_px=5.0, **kw)
    assert set(jax_single.keys()) == set(cpu_single.keys())
    for k_rb in cpu_single:
        np.testing.assert_allclose(
            jax_single[k_rb], cpu_single[k_rb], rtol=2e-3, atol=2e-4
        )

    jax_broad, jax_tight = em._project_and_splat(
        U_true, B, match_sigma_px=12.0, dual_tight_sigma_px=4.5, **kw
    )
    cpu_broad, cpu_tight = em._project_and_splat_cpu(
        U_true, B, match_sigma_px=12.0, dual_tight_sigma_px=4.5, **kw
    )
    for k_rb in cpu_broad:
        np.testing.assert_allclose(
            jax_broad[k_rb], cpu_broad[k_rb], rtol=2e-3, atol=2e-4
        )
        np.testing.assert_allclose(
            jax_tight[k_rb], cpu_tight[k_rb], rtol=2e-3, atol=2e-4
        )


def test_load_observations_shares_prebinned_panel_cache_with_pixel_em(
    tmp_path, monkeypatch
):
    """Single-pass HDF5 slab cache in load_observations matches standalone PixelEMClassifier.from_args."""
    import subhkl.search.pixel_em as pem_mod
    from subhkl.scan_refine import load_observations

    (
        frames_path,
        _boot_path,
        U_true,
        B,
        cell,
        wl,
        d_min,
        _strong_rc,
        _mod_rc,
        _run0_spots,
    ) = _write_synthetic_dataset(tmp_path)

    base_kwargs = {
        "nexus": str(frames_path),
        "instrument": "CG4D",
        "runs": [0, 1, 2],
        "stable_runs": [0, 1, 2],
        "binning": 8,
        "border_width": 4,
        "static_mask_file": None,
        "wl_min": wl[0],
        "wl_max": wl[1],
        "assign_wavelength": wl,
        "d_min": d_min,
        "cell": cell,
        "space_group": "P 2 2 2",
        "finder": None,
    }
    args_cached = SimpleNamespace(**base_kwargs)
    obs = load_observations(args_cached)
    assert list(np.unique(obs[5])) == [0, 1, 2]
    assert getattr(args_cached, "_prebinned_panel_cache", None) is not None

    args_direct = SimpleNamespace(**base_kwargs)
    em_direct = PixelEMClassifier.from_args(args_direct)
    assert em_direct is not None

    def _forbid_reload(*_a, **_kw):
        raise AssertionError("Expected PixelEMClassifier.from_args to hit cache")

    monkeypatch.setattr(pem_mod, "load_binned_nexus_panels", _forbid_reload)
    em_cached = PixelEMClassifier.from_args(args_cached)
    assert em_cached is not None

    em_cached.update_e_step(U_true, B)
    em_direct.update_e_step(U_true, B)
    for bk in em_cached.obs_banks:
        np.testing.assert_allclose(
            em_cached.gamma_valid[bk], em_direct.gamma_valid[bk], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            em_cached.gamma_dark[bk], em_direct.gamma_dark[bk], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            em_cached.gamma_bright[bk], em_direct.gamma_bright[bk], rtol=1e-5, atol=1e-5
        )


def test_band_table_dense_coalescing_matches_exact_quadrature():
    """Dense (>65k) subpixel clouds coalesce cleanly in band_table with >0.999 correlation."""
    from subhkl.config import beamlines
    from subhkl.instrument.detector import Detector
    from subhkl.search.spherical import (
        band_q_range,
        band_table,
        panel_directions,
    )

    rng = np.random.default_rng(77)
    ki = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    all_d, all_w, all_s = [], [], []
    # 6 CG4D panels at binning=4 -> 98,304 > 65,536 bins triggers S^2 x q coalescing
    for bk_key in sorted(beamlines["CG4D"], key=int)[:6]:
        det = Detector(beamlines["CG4D"][bk_key])
        rr, cc = np.meshgrid(
            np.arange(128) * 4 + 1.5, np.arange(128) * 4 + 1.5, indexing="ij"
        )
        dl = panel_directions(det, rows=rr.ravel(), cols=cc.ravel(), ki=ki)
        ex = rng.normal(0.0, 1.0, size=(128, 128))
        # Inject coherent multi-bin Bragg spots (~2-bin width)
        for _ in range(12):
            r0 = int(rng.integers(10, 118))
            c0 = int(rng.integers(10, 118))
            yy, xx = np.ogrid[0:128, 0:128]
            ex += 45.0 * np.exp(-0.5 * ((yy - r0) ** 2 + (xx - c0) ** 2) / 1.8**2)
        all_d.append(dl)
        all_w.append(ex.ravel())
        all_s.append(-(dl @ ki))

    D = np.vstack(all_d).astype(np.float32)
    w = np.concatenate(all_w).astype(np.float32)
    w /= np.abs(w).sum()
    s = np.concatenate(all_s).astype(np.float32)
    qlo, qhi = band_q_range(s, (2.0, 4.5), 1.5)
    qlo_f = qlo.astype(np.float64)
    qhi_f = qhi.astype(np.float64)
    lengths = np.array([8.0, 11.31, 13.86], dtype=np.float32)
    spacing_deg = 8.0

    tab_got, off, nph, nth = band_table(D, w, qlo, qhi, lengths, spacing_deg)
    got_np = np.asarray(tab_got, dtype=np.float64)

    # Independent pure-NumPy exact uncoalesced reference on the Funk pole grid
    th = (np.arange(nth) + 0.5) * np.pi / nth
    off_full = np.concatenate([np.asarray(off, dtype=int), [got_np.shape[1]]])
    nph_np = np.asarray(nph, dtype=int)
    it = np.repeat(np.arange(nth), nph_np)
    ph = (np.arange(off_full[-1]) - np.repeat(off_full[:-1], nph_np) + 0.5) * (
        2.0 * np.pi / nph_np[it]
    )
    P = np.stack(
        [np.sin(th[it]) * np.cos(ph), np.sin(th[it]) * np.sin(ph), np.cos(th[it])], 1
    ).astype(np.float64)

    valid = qhi_f > qlo_f + 1e-7
    D_v = D[valid].astype(np.float64)
    w_v = w[valid].astype(np.float64)
    qlo_v = qlo_f[valid]
    qhi_v = qhi_f[valid]
    env = np.tanh((qhi_v - qlo_v) / np.maximum(0.02 * qlo_v, 1e-6))
    u = P @ D_v.T
    ref_np = np.zeros_like(got_np)
    two_pi = 2.0 * np.pi
    for iz, xz in enumerate(lengths.astype(np.float64)):
        t = xz * u
        den = two_pi * (qhi_v[None, :] - qlo_v[None, :]) * t
        ok = np.abs(den) > 1e-7
        num = np.sin(two_pi * qhi_v[None, :] * t) - np.sin(two_pi * qlo_v[None, :] * t)
        k_val = env[None, :] * np.where(ok, num / np.where(ok, den, 1.0), 1.0)
        ref_np[iz] = k_val @ w_v

    corr = float(np.corrcoef(ref_np.ravel(), got_np.ravel())[0, 1])
    rel_err = float(np.linalg.norm(ref_np - got_np) / np.linalg.norm(ref_np))
    assert corr > 0.999
    assert rel_err < 0.03
    # band_phase is antipodal-symmetric (phi(-p) == phi(+p)), so compare |p_got . p_ref|
    p_got = P[int(np.argmax(got_np[0]))]
    p_ref = P[int(np.argmax(ref_np[0]))]
    assert abs(float(np.dot(p_got, p_ref))) > 0.999


def test_peer_ring_backscattering_shadow_masks_whole_panel():
    """Cross-bank same-2theta peer ring catches a whole-panel shadow (~0.30x peers) without false positives."""
    rng = np.random.default_rng(36185)
    # 8 MANDI banks including high-angle backscattering ring peers around 2theta ~ 140-160 deg
    banks = [10, 13, 20, 22, 23, 33, 42, 53]
    runs = [0, 1, 2, 3]
    hb = wb = 32
    y_rb, good_rb = {}, {}
    for r in runs:
        for bk in banks:
            rate = 400.0 if bk == 33 else (820.0 if bk == 23 else 1500.0)
            y_rb[(r, bk)] = rng.poisson(rate, size=(hb, wb)).astype(np.float64)
            good_rb[(r, bk)] = np.ones((hb, wb), dtype=bool)

    clf = PixelEMClassifier(
        instrument="MANDI",
        n_rows=256,
        n_cols=256,
        binning=8,
        border_width=15,
        sel_runs=runs,
        obs_banks=banks,
        y_rb=y_rb,
        good_rb=good_rb,
        rr_rb={},
        cc_rb={},
        bad_px_bank={},
        R_by_run={r: np.eye(3) for r in runs},
        cell=[55.16, 81.19, 88.7, 90.0, 96.3, 90.0],
        space_group="I 1 2 1",
        wavelength=(2.0, 4.0),
        d_min=2.3,
    )
    assert float((clf.gamma_dark[33] > 0.5).mean()) > 0.98
    assert float((clf.gamma_dark[23] > 0.5).mean()) > 0.90
    assert float((clf.gamma_dark[20] > 0.5).mean()) == 0.0
