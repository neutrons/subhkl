"""Static-structure masks: estimation, bank mapping, and finder semantics.

The finder test is the contract that matters: masked pixels are missing
data, not zeroed counts, so detections vanish from the masked region while
a real peak elsewhere in the frame keeps its position and flux.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.special import erf

from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder
from subhkl.search.static_mask import (
    build_mask_file,
    estimate_static_mask,
    load_mask_for_banks,
)


def pixel_integrated_gaussian(shape, r0, c0, sigma, amplitude):
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    s2 = sigma * np.sqrt(2.0)
    er = erf((rr - r0 + 0.5) / s2) - erf((rr - r0 - 0.5) / s2)
    ec = erf((cc - c0 + 0.5) / s2) - erf((cc - c0 - 0.5) / s2)
    return amplitude * (np.pi / 2.0) * sigma**2 * er * ec


def synthetic_scan(n_frames=12, size=128, step_col=80, glow=None, peaks=True, seed=5):
    """Frames with a static illumination step, optional static glow, and a
    bright peak that moves frame to frame -- the l1-mbl artifact anatomy."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        rate = np.full((size, size), 2.2)
        rate[:, step_col:] = 0.6  # illumination boundary, static
        if glow is not None:
            rr, cc = np.mgrid[0:size, 0:size]
            r0, c0, rad, level = glow
            rate += level * ((rr - r0) ** 2 + (cc - c0) ** 2 < rad**2)
        if peaks:
            rate += pixel_integrated_gaussian(
                rate.shape, 20.0 + 7 * k, 15.0 + 4 * k, 2.0, 80.0
            )
        frames.append(rng.poisson(rate).astype(np.float32))
    return np.stack(frames)


def test_static_mask_catches_the_boundary_and_glow_but_not_the_moving_peak():
    frames = synthetic_scan(glow=(100, 30, 10, 2.5))
    valid = estimate_static_mask(frames, dilate_px=4)

    # The illumination step is masked...
    assert valid[:, 76:84].mean() < 0.1
    # ...and so is the static glow.
    assert valid[92:108, 22:38].mean() < 0.2
    # The moving peak's positions stay valid: the median never saw it.
    for k in range(12):
        assert valid[20 + 7 * k, 15 + 4 * k] == 1
    # And the bulk of the panel is untouched.
    assert valid.mean() > 0.7


def test_static_mask_needs_at_least_two_frames():
    with pytest.raises(ValueError):
        estimate_static_mask(np.zeros((1, 32, 32)))


def test_mask_file_round_trip_maps_by_physical_bank(tmp_path):
    # Moving peaks keep the frames genuinely distinct scenes: peak-free
    # re-draws of one static pattern are re-exposures, and the duplicate
    # guard rightly collapses those.
    frames_a = synthetic_scan(n_frames=6, size=64, step_col=40, seed=1)
    frames_b = synthetic_scan(n_frames=6, size=64, step_col=20, seed=2)

    # Two input files, interleaved banks, on purpose out of order.
    for name, banks, stacks in (
        ("scan1.h5", [7, 3], (frames_a[:3], frames_b[:3])),
        ("scan2.h5", [3, 7], (frames_b[3:], frames_a[3:])),
    ):
        with h5py.File(tmp_path / name, "w") as f:
            f["images"] = np.concatenate(stacks)
            f["bank_ids"] = np.repeat(banks, 3)

    out = tmp_path / "mask.h5"
    # wide_sigma pinned to the fixture scale: a 64 px canvas cannot host the
    # 20 px default (the wide smooth degenerates to the global mean and the
    # plateau inflates the MAD floor) -- the subject here is bank mapping.
    summary = build_mask_file(
        [tmp_path / "scan1.h5", tmp_path / "scan2.h5"],
        out,
        min_frames=5,
        dilate_px=2,
        wide_sigma=10.0,
    )
    assert summary["banks"] == [3, 7]
    assert summary["n_frames"] == {3: 6, 7: 6}
    assert summary["thin_banks"] == []

    # Bank 3's step is at column 20, bank 7's at column 40; the mapping must
    # place each mask with its own bank whatever the requested order, and a
    # bank the file does not carry comes back fully valid.  On this small
    # fixture the strip is soft (sharpness is covered by the 128 px tests),
    # so the assertion is the mapping: each bank is masked at *its own*
    # step's bright shoulder and clean at the other bank's.
    stack = load_mask_for_banks(out, [7, 99, 3], (64, 64))
    assert stack.shape == (3, 64, 64)
    assert stack[0][:, 32:40].mean() < 0.6 and stack[0][:, 12:20].mean() > 0.9
    assert stack[2][:, 12:20].mean() < 0.6 and stack[2][:, 32:40].mean() > 0.9
    assert stack[1].min() == 1.0

    with pytest.raises(ValueError):
        load_mask_for_banks(out, [3], (32, 32))


def test_mask_file_keeps_thin_banks_fully_valid(tmp_path):
    frames = synthetic_scan(n_frames=3, size=64, peaks=False)
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = frames
        f["bank_ids"] = np.full(3, 11)
    summary = build_mask_file(
        [tmp_path / "scan.h5"], tmp_path / "mask.h5", min_frames=5
    )
    assert summary["thin_banks"] == [11]
    assert load_mask_for_banks(tmp_path / "mask.h5", [11], (64, 64)).min() == 1.0


def test_finder_honors_the_mask_without_touching_the_counts():
    """An illumination step manufactures detections along its ridge; the mask
    removes them as missing data while the real peak keeps its flux."""
    rng = np.random.default_rng(9)
    size, step_col = 96, 60
    rate = np.full((size, size), 2.2)
    rate[:, step_col:] = 0.6
    rate += pixel_integrated_gaussian(rate.shape, 30.0, 25.0, 2.0, 120.0)
    image = rng.poisson(rate).astype(np.float32)
    stack = np.stack([image, image])

    def run(valid):
        finder = MatrixFreeSparseRBFPeakFinder(
            min_sigma=1.5,
            max_sigma=3.0,
            num_sigmas=3,
            profile_file="gaussian",
            shape_ratio=1.0,
        )
        return finder.find_peaks_batch(stack, valid=valid)

    unmasked = np.asarray(run(None)[0])

    valid = np.ones_like(stack)
    valid[:, :, step_col - 8 : step_col + 8] = 0.0
    masked = np.asarray(run(valid)[0])

    def near_step(peaks):
        return peaks[np.abs(peaks[:, 2] - step_col) < 8]

    def real_peak(peaks):
        sel = (np.abs(peaks[:, 1] - 30) < 3) & (np.abs(peaks[:, 2] - 25) < 3)
        return peaks[sel]

    # The step ridge attracts detections when unmasked -- that is the bug
    # being tested -- and none survive inside the mask.
    assert len(near_step(unmasked)) >= 1
    assert len(near_step(masked)) == 0

    # The real peak is found either way, at the same place and flux: masking
    # is missing data, not a modification of the counted image.
    a, b = real_peak(unmasked), real_peak(masked)
    assert len(a) == 1 and len(b) == 1
    np.testing.assert_allclose(a[0][1:3], b[0][1:3], atol=0.3)
    np.testing.assert_allclose(a[0][0], b[0][0], rtol=0.05)


def test_fully_valid_mask_is_the_unmasked_path():
    """A mask of ones must not even change the trace, let alone the answer."""
    rng = np.random.default_rng(3)
    image = rng.poisson(np.full((64, 64), 1.0)).astype(np.float32)
    image += pixel_integrated_gaussian(image.shape, 32.0, 32.0, 2.0, 100.0)
    stack = image[None]
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.5,
        max_sigma=3.0,
        num_sigmas=3,
        profile_file="gaussian",
        shape_ratio=1.0,
    )
    plain = np.asarray(finder.find_peaks_batch(stack)[0])
    ones = np.asarray(finder.find_peaks_batch(stack, valid=np.ones_like(stack))[0])
    np.testing.assert_array_equal(plain, ones)


def test_mask_is_invariant_to_exposure_time():
    """The same static feature at 5x the counts must give the same mask:
    frames are normalised to unit mean rate, and both criteria are ratios."""
    short = synthetic_scan(peaks=False, seed=4)
    long_ = np.stack(
        [
            np.random.default_rng(40 + k)
            .poisson(5.0 * (2.2 - 1.6 * (np.arange(128) >= 80)))
            .astype(np.float32)[None, :]
            .repeat(128, axis=0)
            for k in range(12)
        ]
    )
    # Rebuild the long exposure properly: same rate pattern as synthetic_scan,
    # scaled 5x, fresh Poisson draws.
    rng = np.random.default_rng(21)
    rate = np.full((128, 128), 2.2)
    rate[:, 80:] = 0.6
    long_ = np.stack([rng.poisson(5.0 * rate).astype(np.float32) for _ in range(12)])

    a = estimate_static_mask(short, dilate_px=4)
    b = estimate_static_mask(long_, dilate_px=4)
    # Same masked geometry.  The masked set is a ~20-column strip, so a
    # couple of columns of edge wiggle between independent noise draws
    # costs ~10-15% of Jaccard overlap; 0.75 asserts same-geometry while
    # tolerating that wiggle.
    both, either = ((a == 0) & (b == 0)).sum(), ((a == 0) | (b == 0)).sum()
    assert both / either > 0.75

    # And mixing the two exposures in one stack does not change the answer.
    mixed = estimate_static_mask(np.concatenate([short, long_]), dilate_px=4)
    both, either = ((mixed == 0) & (a == 0)).sum(), ((mixed == 0) | (a == 0)).sum()
    assert both / either > 0.75


def test_dense_diffraction_is_protected_and_subset_artifacts_are_caught():
    """The two sides of the quantile union's bargain, in one frame stack.

    A reflection ring whose positions recur every few frames leaks into the
    p75 map by the same token that makes the map see subset-frame
    artifacts; the pipeline's answer is certificates, so the ring's
    (certified) positions must stay valid through protection.  Meanwhile an
    artifact present in only a third of the frames -- an illumination
    boundary that shifts with a manually-set orientation class, invisible
    to the low quantile by construction -- must be caught by the high one.
    The always-static edge stays masked on its bright shoulder, and only
    there: false peaks are positive structure."""
    rng = np.random.default_rng(17)
    size = 128
    ring_pos = []
    for j in range(10):
        angle = 2 * np.pi * j / 10
        ring_pos.append((60.0 + 25 * np.sin(angle), 38.0 + 25 * np.cos(angle)))
    frames = []
    for k in range(12):
        rate = np.full((size, size), 1.0)
        rate[:, 90:] = 0.3  # always-static edge
        if k < 4:
            # The subset artifact: a bright strip present in 4 of 12 frames
            # (kept clear of the ring so no protection disk overlaps it).
            rate[2:8, :] = 3.5
        for j, (r0, c0) in enumerate(ring_pos):
            if (k + j) % 6:
                continue
            rate += pixel_integrated_gaussian(
                rate.shape,
                r0 + rng.integers(-2, 3),
                c0 + rng.integers(-2, 3),
                2.0,
                60.0,
            )
        frames.append(rng.poisson(rate).astype(np.float32))

    disks = [(r0, c0, 2.0) for r0, c0 in ring_pos]
    valid = estimate_static_mask(np.stack(frames), dilate_px=4, protect_disks=disks)

    # The static edge's bright shoulder is masked; the dark side is not.
    assert valid[:, 84:90].mean() < 0.2
    # The subset-frame strip is masked: the p75 map holds it.  (The
    # interior; the outermost rows see border-weakened band estimates.)
    assert valid[3:8, 20:70].mean() < 0.2
    # The certified ring stays findable through protection.
    rr, cc = np.mgrid[0:size, 0:size]
    ring = np.abs(np.hypot(rr - 60.0, cc - 38.0) - 25.0) < 6
    assert valid[ring].mean() > 0.9


def test_a_wide_smooth_halo_is_not_masked_and_its_peak_survives():
    """The background model follows broad structure, so the mask must not
    claim it: absolute-level and absolute-gradient criteria removed genuine
    Bragg peaks from the centre of the l1-mbl forward banks."""
    rng = np.random.default_rng(23)
    size = 128
    rr, cc = np.mgrid[0:size, 0:size]
    halo = 3.0 * np.exp(-((rr - 64.0) ** 2 + (cc - 64.0) ** 2) / (2 * 40.0**2))
    frames = []
    for k in range(12):
        rate = 1.0 + halo
        rate[:, 100:] = 0.3  # a real step stays in frame for contrast
        frames.append(rng.poisson(rate).astype(np.float32))
    valid = estimate_static_mask(np.stack(frames), dilate_px=4)

    # The halo centre -- 4x ambient, statically steep shoulders -- is valid.
    assert valid[44:84, 44:84].mean() > 0.95
    # The genuine step is still masked.
    assert valid[:, 96:104].mean() < 0.2


def test_repeated_frames_do_not_promote_signal_into_the_mask(tmp_path):
    """The same frame (or the same goniometer orientation re-exposed) fed
    repeatedly would make the true signal static by construction.  Repeats
    are dropped, reported, and a bank left with too few distinct frames
    stays fully valid."""
    rng = np.random.default_rng(31)
    rate = np.full((64, 64), 1.0)
    rate += pixel_integrated_gaussian(rate.shape, 30.0, 30.0, 2.0, 200.0)
    frame = rng.poisson(rate).astype(np.float32)

    # Same content ten times over (no goniometer bookkeeping in the file).
    with h5py.File(tmp_path / "dup.h5", "w") as f:
        f["images"] = np.stack([frame] * 10)
        f["bank_ids"] = np.full(10, 5)
    summary = build_mask_file([tmp_path / "dup.h5"], tmp_path / "m1.h5", min_frames=5)
    assert summary["duplicates_dropped"] == {5: 9}
    assert summary["thin_banks"] == [5]
    assert load_mask_for_banks(tmp_path / "m1.h5", [5], (64, 64)).min() == 1.0

    # Re-exposures: same angles, fresh Poisson draws.  Content differs, the
    # orientation does not, and the goniometer record is what catches it.
    with h5py.File(tmp_path / "reexp.h5", "w") as f:
        f["images"] = np.stack(
            [rng.poisson(rate).astype(np.float32) for _ in range(10)]
        )
        f["bank_ids"] = np.full(10, 5)
        f["goniometer/angles"] = np.tile([10.0, 0.0, 0.0], (10, 1))
    summary = build_mask_file([tmp_path / "reexp.h5"], tmp_path / "m2.h5", min_frames=5)
    assert summary["duplicates_dropped"] == {5: 9}
    assert load_mask_for_banks(tmp_path / "m2.h5", [5], (64, 64)).min() == 1.0


def test_no_exclusion_ring_around_a_static_hot_spot():
    """A static bright spot must mask its own footprint and nothing more:
    the |band-pass| criterion dug a ~wide-sigma moat around anything bright,
    which on real forward banks became a ~20 px exclusion zone around every
    leaked reflection."""
    rng = np.random.default_rng(41)
    size = 128
    frames = []
    for _ in range(12):
        rate = np.full((size, size), 1.0)
        rate += pixel_integrated_gaussian(rate.shape, 64.0, 64.0, 2.5, 50.0)
        frames.append(rng.poisson(rate).astype(np.float32))
    valid = estimate_static_mask(np.stack(frames), dilate_px=4)

    rr, cc = np.mgrid[0:size, 0:size]
    radius = np.hypot(rr - 64.0, cc - 64.0)
    # Core masked (it is static structure), annulus untouched.
    assert valid[radius < 4].mean() < 0.2
    assert valid[(radius > 16) & (radius < 28)].mean() > 0.95


def test_metric_certified_peaks_are_exonerated_and_artifacts_are_not(tmp_path):
    """The user-facing contract of the peaks input: a quasi-static genuine
    reflection (manually repeated Laue orientations) is rescued by its fit
    metrics, while a detection on the illumination step -- poor evidence,
    poor shape -- earns no exoneration and its structure stays masked."""
    rng = np.random.default_rng(53)
    size, step_col = 128, 80
    sigma_pk = 2.0
    peak = pixel_integrated_gaussian((size, size), 40.0, 30.0, sigma_pk, 800.0)
    frames = []
    for _ in range(10):
        rate = np.full((size, size), 2.2)
        rate[:, step_col:] = 0.6
        # The same bright reflection in EVERY frame: static by any statistic.
        rate += peak
        frames.append(rng.poisson(rate).astype(np.float32))

    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = np.stack(frames)
        f["bank_ids"] = np.full(10, 4)
        # Distinct nominal angles so the duplicate guard sees a moving scan.
        f["goniometer/angles"] = np.c_[np.arange(10.0), np.zeros(10)]

    def write_peaks(path, deviance, residual):
        with h5py.File(path, "w") as f:
            n = 10
            f["peaks/image_index"] = np.arange(n)
            f["peaks/pixel_r"] = np.full(n, 40.0)
            f["peaks/pixel_c"] = np.full(n, 30.0)
            f["peaks/sigma"] = np.full(n, sigma_pk)
            f["peaks/deviance"] = np.full(n, deviance)
            f["peaks/residual_deviance"] = np.full(n, residual)

    # Without exoneration the persistent reflection is (correctly, by the
    # statistics alone) static -- and masked.  This is the hazard.
    from subhkl.search.static_mask import build_mask_file, load_mask_for_banks

    # wide_sigma pinned to the fixture scale (see the bank-mapping test);
    # the subject here is the exoneration contract.
    build_mask_file(
        [tmp_path / "scan.h5"], tmp_path / "m0.h5", dilate_px=4, wide_sigma=10.0
    )
    m0 = load_mask_for_banks(tmp_path / "m0.h5", [4], (size, size))[0]
    assert m0[36:44, 26:34].mean() < 0.5

    # Confident metrics rescue it; the step stays masked.
    write_peaks(tmp_path / "peaks.h5", deviance=120.0, residual=1.1)
    build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m1.h5",
        peaks=[tmp_path / "peaks.h5"],
        dilate_px=4,
        wide_sigma=10.0,
    )
    m1 = load_mask_for_banks(tmp_path / "m1.h5", [4], (size, size))[0]
    assert m1[36:44, 26:34].mean() > 0.9
    # No ring either: the amplitude-aware protection covers the bright tail
    # AND survives the dilation of the bad set, so the whole footprint --
    # centre, skirt and margin -- stays findable.  The evidence clearance
    # alone left a masked annulus here (the tail exceeds the texture
    # threshold beyond 3 sigma for a peak this bright, and dilation grew it
    # back over the cleared core).
    rr_t, cc_t = np.mgrid[0:size, 0:size]
    footprint = np.hypot(rr_t - 40.0, cc_t - 30.0) <= 14.0
    assert m1[footprint].mean() > 0.98
    assert m1[:, 74:80].mean() < 0.2

    # A faint peak's metrics also rescue it: deviance 20 clears the chi^2_4
    # admission level (9.49) even though it would have failed the old bar of
    # 25.  Measured on l1-mbl: 340 detections sat at deviance 20-23 with
    # clean shape -- bright enough to mask, too faint for a deviance-25
    # certificate, so they were masked with no route to exoneration.
    write_peaks(tmp_path / "faint.h5", deviance=20.0, residual=1.1)
    build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m3.h5",
        peaks=[tmp_path / "faint.h5"],
        dilate_px=4,
        wide_sigma=10.0,
    )
    m3 = load_mask_for_banks(tmp_path / "m3.h5", [4], (size, size))[0]
    assert m3[36:44, 26:34].mean() > 0.9
    assert m3[:, 74:80].mean() < 0.2
    with h5py.File(tmp_path / "m3.h5", "r") as f:
        assert f.attrs["peak_deviance_min"] == pytest.approx(9.488)
        assert f.attrs["n_exonerated"] == 10
        assert list(f.attrs["peaks"]) == [str(tmp_path / "faint.h5")]

    # Poor metrics -- an artifact's -- rescue nothing.
    write_peaks(tmp_path / "bad.h5", deviance=6.0, residual=3.5)
    build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m2.h5",
        peaks=[tmp_path / "bad.h5"],
        dilate_px=4,
        wide_sigma=10.0,
    )
    m2 = load_mask_for_banks(tmp_path / "m2.h5", [4], (size, size))[0]
    assert m2[36:44, 26:34].mean() < 0.5


def test_no_annulus_around_a_quasi_static_certified_peak():
    """Exoneration must not manufacture the artifact it protects against.

    A certified peak that barely moves between frames (a Laue zone rotating
    about its own axis) writes its whole footprint into the static map.  An
    earlier design cleared the certified evidence out of the frame stack,
    and the clearing itself manufactured a masked annulus around the peak
    (the no-evidence crater dug a positive band-pass rim just outside
    itself).  Certificates now only protect, with a radius measured off the
    estimator's own quantile maps, covering wherever the peak's smoothed
    tail exceeds the texture threshold -- so nothing near the peak may be masked, at any radius."""
    rng = np.random.default_rng(67)
    size, sigma_pk, height = 160, 2.0, 200.0
    wobble = [
        (0, 0), (1, -1), (-1, 1), (2, 0), (-2, 2), (0, -2),
        (1, 1), (-1, -2), (2, -2), (0, 1), (-2, 0), (1, 2),
    ]  # fmt: skip
    frames, disks = [], []
    for dr, dc in wobble:
        rate = np.full((size, size), 5.0)
        rate += pixel_integrated_gaussian(
            rate.shape, 64.0 + dr, 64.0 + dc, sigma_pk, height
        )
        frames.append(rng.poisson(rate).astype(np.float32))
        disks.append((64.0 + dr, 64.0 + dc, sigma_pk))
    valid = estimate_static_mask(np.stack(frames), protect_disks=disks)

    rr, cc = np.mgrid[0:size, 0:size]
    radius = np.hypot(rr - 64.0, cc - 64.0)
    # No annulus at any radius: core, skirt and far field all stay valid.
    assert valid[radius < 40].mean() > 0.995
    assert valid.mean() > 0.98


def test_a_faint_coherent_ridge_is_masked_contiguously():
    """A ridge whose contrast clears the boundary criterion is caught along
    its whole length, at any orientation: the smoothed gradient magnitude
    consolidates the feature's flanks, so pointwise noise dips do not
    puncture the mask.  The moving peak's positions stay valid and the far
    field stays clean -- the criterion responds to coherent contrast, not
    to noise."""
    rng = np.random.default_rng(71)
    size = 128
    cols = np.arange(size, dtype=float)
    ridge = 0.8 * np.exp(-0.5 * ((cols - 60.0) / 2.0) ** 2)
    frames = []
    for k in range(10):
        rate = np.full((size, size), 2.2) + ridge[None, :]
        rate += pixel_integrated_gaussian(
            (size, size), 12.0 + 10 * k, 100.0 + 2 * k, 2.0, 60.0
        )
        frames.append(rng.poisson(rate).astype(np.float32))

    valid = estimate_static_mask(np.stack(frames))
    col = slice(58, 63)
    assert (valid[:, col] == 0).mean() > 0.999
    assert (valid[:, 90:] == 0).mean() < 0.01
    for k in range(10):
        assert valid[12 + 10 * k, 100 + 2 * k] == 1


def test_summed_file_pools_each_banks_evidence(tmp_path):
    from subhkl.search.static_mask import build_summed_file

    frames_a = synthetic_scan(n_frames=4, size=64, step_col=40, seed=1)
    frames_b = synthetic_scan(n_frames=4, size=64, step_col=20, seed=2)
    for name, banks, stacks in (
        ("scan1.h5", [7, 3], (frames_a[:2], frames_b[:2])),
        ("scan2.h5", [3, 7], (frames_b[2:], frames_a[2:])),
    ):
        with h5py.File(tmp_path / name, "w") as f:
            f["images"] = np.concatenate(stacks)
            f["bank_ids"] = np.repeat(banks, 2)
            f.attrs["instrument"] = "CG4D"
            f["goniometer/angles"] = np.arange(8.0).reshape(4, 2)

    summary = build_summed_file(
        [tmp_path / "scan1.h5", tmp_path / "scan2.h5"], tmp_path / "summed.h5"
    )
    assert summary["banks"] == [3, 7]
    assert summary["n_frames"] == {3: 4, 7: 4}

    with h5py.File(tmp_path / "summed.h5", "r") as f:
        assert list(f["bank_ids"][()]) == [3, 7]
        # The sum is exact per bank, whatever order the files carried them in.
        np.testing.assert_allclose(f["images"][0], frames_b.sum(axis=0))
        np.testing.assert_allclose(f["images"][1], frames_a.sum(axis=0))
        assert f.attrs["instrument"] == "CG4D"
        assert list(f.attrs["n_frames"]) == [4, 4]
        # Placeholder angles, one per bank frame: metrics only, never indexing.
        assert f["goniometer/angles"].shape == (2, 2)
        assert np.all(f["goniometer/angles"][()] == 0.0)


def test_pooled_peaks_rescue_a_subthreshold_static_peak(tmp_path):
    """The compounding contract: a quasi-static reflection too faint for any
    single frame's certificate is masked by the pooled static map -- so its
    exoneration must come from the same pooling.  A finder run on the summed
    stack certifies it (deviance is additive), and its per-peak bank routes
    the protection into every frame of that bank, with the pooled amplitude
    rescaled to the per-frame one."""
    rng = np.random.default_rng(59)
    size, step_col, sigma_pk = 128, 80, 2.0
    n_frames = 10
    # Faint: ~55 counts of flux -- per-frame deviance ~ flux^2/(2 pi sigma^2
    # * 4 * bg) sits well under any admission level, yet the smoothed p25
    # holds ~0.6 counts/px against an ambient of 2.2: masked.
    peak = pixel_integrated_gaussian(
        (size, size), 40.0, 30.0, sigma_pk, 55.0 / (2 * np.pi * sigma_pk**2)
    )
    frames = []
    for _ in range(n_frames):
        rate = np.full((size, size), 2.2)
        rate[:, step_col:] = 0.6
        rate += peak
        frames.append(rng.poisson(rate).astype(np.float32))
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = np.stack(frames)
        f["bank_ids"] = np.full(n_frames, 4)
        f["goniometer/angles"] = np.c_[np.arange(float(n_frames)), np.zeros(n_frames)]

    # Without rescue the faint static peak is masked.
    build_mask_file([tmp_path / "scan.h5"], tmp_path / "m0.h5", dilate_px=4)
    m0 = load_mask_for_banks(tmp_path / "m0.h5", [4], (size, size))[0]
    assert m0[38:43, 28:33].mean() < 0.5

    # The pooled certificate: one detection, bank-addressed, with the summed
    # flux and a deviance only the pooled fit could reach.  Residual 8.0 is
    # deliberate: goodness of fit gets stricter with counts (its mismatch
    # component scales with them), so a genuine peak's per-frame ~1.1 lands
    # well above a naive bar of 2 on the tenfold sum.  The scaled bar
    # 1 + (residual_max - 1) * n = 11 must admit it.
    with h5py.File(tmp_path / "pooled.h5", "w") as f:
        f["bank"] = np.array([4])
        f["peaks/image_index"] = np.array([0])
        f["peaks/pixel_r"] = np.array([40.0])
        f["peaks/pixel_c"] = np.array([30.0])
        f["peaks/sigma"] = np.array([sigma_pk])
        f["peaks/deviance"] = np.array([60.0])
        f["peaks/residual_deviance"] = np.array([8.0])

    summary = build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m1.h5",
        pooled_peaks=tmp_path / "pooled.h5",
        dilate_px=4,
    )
    assert summary["n_exonerated_pooled"] == 1
    m1 = load_mask_for_banks(tmp_path / "m1.h5", [4], (size, size))[0]
    # The peak footprint is protected; the illumination step stays masked.
    assert m1[36:44, 26:34].mean() > 0.9
    assert m1[:, 74:80].mean() < 0.2
    with h5py.File(tmp_path / "m1.h5", "r") as f:
        assert f.attrs["pooled_peaks"] == str(tmp_path / "pooled.h5")
        assert f.attrs["n_exonerated_pooled"] == 1

    # A pooled certificate with an artifact's shape rescues nothing: a
    # per-frame residual of ~3.5 scales to ~1 + 2.5 n = 26 on the sum,
    # above the scaled bar of 11.
    with h5py.File(tmp_path / "pooled_bad.h5", "w") as f:
        f["bank"] = np.array([4])
        f["peaks/image_index"] = np.array([0])
        f["peaks/pixel_r"] = np.array([40.0])
        f["peaks/pixel_c"] = np.array([30.0])
        f["peaks/sigma"] = np.array([sigma_pk])
        f["peaks/deviance"] = np.array([60.0])
        f["peaks/residual_deviance"] = np.array([26.0])
    build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m2.h5",
        pooled_peaks=tmp_path / "pooled_bad.h5",
        dilate_px=4,
    )
    m2 = load_mask_for_banks(tmp_path / "m2.h5", [4], (size, size))[0]
    assert m2[38:43, 28:33].mean() < 0.5


def test_a_certificate_on_extended_structure_is_refused(tmp_path):
    """Exoneration is a statement about a peak, not the structure it sits on.

    Detections along the illumination edge carry clean metrics -- measured
    on l1-mbl at deviance 20+, residual/DoF < 2, indistinguishable from
    faint genuine peaks -- and at the admission-level bar each would clear
    an evidence crater and punch a protection disk; a chain of them
    dissolves the edge mask.  The gate is geometric: a certificate explains
    at most the peak's own amplitude-aware footprint, so a detection whose
    underlying static component extends beyond 4x the protected radius is
    refused, while the genuine compact peak next to it keeps its rescue."""
    rng = np.random.default_rng(61)
    size, step_col, sigma_pk = 128, 80, 2.0
    peak = pixel_integrated_gaussian((size, size), 40.0, 30.0, sigma_pk, 800.0)
    frames = []
    for _ in range(10):
        rate = np.full((size, size), 2.2)
        rate[:, step_col:] = 0.6
        rate += peak
        frames.append(rng.poisson(rate).astype(np.float32))
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = np.stack(frames)
        f["bank_ids"] = np.full(10, 4)
        f["goniometer/angles"] = np.c_[np.arange(10.0), np.zeros(10)]

    # Every frame certifies BOTH the genuine peak and a detection sitting on
    # the illumination edge, with identical (clean) metrics.
    with h5py.File(tmp_path / "peaks.h5", "w") as f:
        n = 20
        f["peaks/image_index"] = np.repeat(np.arange(10), 2)
        f["peaks/pixel_r"] = np.tile([40.0, 64.0], 10)
        f["peaks/pixel_c"] = np.tile([30.0, float(step_col)], 10)
        f["peaks/sigma"] = np.full(n, sigma_pk)
        f["peaks/deviance"] = np.full(n, 120.0)
        f["peaks/residual_deviance"] = np.full(n, 1.1)

    # wide_sigma pinned to the fixture scale; the subject here is the gate.
    build_mask_file(
        [tmp_path / "scan.h5"],
        tmp_path / "m.h5",
        peaks=[tmp_path / "peaks.h5"],
        dilate_px=4,
        wide_sigma=10.0,
    )
    m = load_mask_for_banks(tmp_path / "m.h5", [4], (size, size))[0]
    # The genuine peak keeps its rescue ...
    assert m[36:44, 26:34].mean() > 0.9
    # ... and the edge keeps its mask, including where the refused
    # certificate would have cleared and protected it.
    assert m[:, 76:80].mean() < 0.2
    assert m[56:72, 76:80].mean() < 0.2


def test_peaks_must_pair_with_inputs(tmp_path):
    with h5py.File(tmp_path / "a.h5", "w") as f:
        f["images"] = np.zeros((2, 16, 16), dtype=np.float32)
        f["bank_ids"] = np.array([1, 1])
    with pytest.raises(ValueError, match="pair with the inputs"):
        build_mask_file([tmp_path / "a.h5"], tmp_path / "m.h5", peaks=[])


def test_mask_file_records_the_instrument_its_inputs_knew(tmp_path):
    frames = synthetic_scan(n_frames=6, size=64)
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = frames
        f["bank_ids"] = np.full(6, 3)
        f["goniometer/angles"] = np.c_[np.arange(6.0), np.zeros(6)]
        f.attrs["instrument"] = "CG4D"
    build_mask_file([tmp_path / "scan.h5"], tmp_path / "mask.h5")
    with h5py.File(tmp_path / "mask.h5") as f:
        assert f.attrs["instrument"] == "CG4D"
