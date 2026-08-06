"""Overlap regression cases for the peak finder.

These two document failure modes of the greedy matching-pursuit finder: a pair
of heavily overlapping peaks reported as one composite "ghost" in the middle,
and a weak peak in the tail of a strong one lost to over-subtraction.  Their
original docstrings note the fix would be "OMP or Joint Optimization", which is
what the matrix-free global basis pursuit finder is, so they now exercise that.

They run at ``gamma=0.5``; ``gamma=1`` is the degenerate value at which the
penalty cannot prefer one broad atom to several overlapping ones.  See
docs/matrix_free_theory.md Theorem 1.
"""

import numpy as np
import scipy.special


def generate_erf_peak(y_coords, x_coords, r, c, sig, amp):
    """
    Helper function to generate physically exact subpixel peaks
    using the continuous analytic Gaussian pixel integral.
    """
    sig_sq2 = sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf(
        (y_coords - 0.5 - r) / sig_sq2
    )
    erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf(
        (x_coords - 0.5 - c) / sig_sq2
    )
    return amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x


def test_overlapping_ghost_center_shift_failure():
    """
    EXPOSES: Ghost Center Shifts / Mutual Energy Swapping.
    Two closely spaced, heavily overlapping peaks should be detected as two distinct centers.
    A naive greedy algorithm will detect a single composite "ghost" peak in the middle,
    leaving artificial residuals.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 100, 100
    bg_level = 10.0
    np.random.seed(42)

    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Inject two true peaks physically overlapping to form a single blended maximum
    true_r = 50.0
    center_1, center_2 = 42.0, 58.0

    # Large sigmas (6.0) ensure the spatial domains overlap heavily
    image += generate_erf_peak(y_coords, x_coords, true_r, center_1, sig=6.0, amp=50.0)
    image += generate_erf_peak(y_coords, x_coords, true_r, center_2, sig=6.0, amp=50.0)

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=2.0,
        max_sigma=8.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]  # [intensity, r, c, sigma]

    extracted_c = peaks[:, 2]

    # --- ASSERTIONS ---
    # These define the EXPECTED correct behavior and will expose the greedy solver's failure mode.

    # 1. It should find at least 2 distinct peaks
    assert len(extracted_c) >= 2, (
        "Algorithm merged overlapping peaks into a single detection."
    )

    # 2. It should accurately isolate the two true centers
    assert any(abs(c - center_1) < 2.0 for c in extracted_c), (
        "Failed to isolate the Left peak."
    )
    assert any(abs(c - center_2) < 2.0 for c in extracted_c), (
        "Failed to isolate the Right peak."
    )

    # 3. It MUST NOT hallucinate a ghost peak in the exact middle of the composite blob
    ghost_zone = [c for c in extracted_c if 48.0 < c < 52.0]
    assert not ghost_zone, (
        f"Failure Mode Exposed: Algorithm hallucinated a ghost peak at the composite center: {ghost_zone}"
    )


def test_oversparsification_missing_weak_peak_failure():
    """
    EXPOSES: Over-Sparsification / Missing Weak Overlapping Peaks.
    A very strong peak overlaps a weak peak. The weak peak's intensity
    is often over-subtracted or shifted below the detection threshold
    when the dominant peak is greedily evaluated first.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 100, 100
    bg_level = 10.0
    np.random.seed(101)

    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    true_r = 50.0
    strong_c = 30.0
    weak_c = 38.0

    # 1. Inject strong primary peak
    image += generate_erf_peak(y_coords, x_coords, true_r, strong_c, sig=4.0, amp=200.0)
    # 2. Inject weak secondary peak hidden in the right tail
    image += generate_erf_peak(y_coords, x_coords, true_r, weak_c, sig=2.0, amp=30.0)

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=1.0,
        max_sigma=6.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    extracted_c = peaks[:, 2]

    # --- ASSERTIONS ---
    strong_peak_detected = any(abs(c - strong_c) < 1.5 for c in extracted_c)
    weak_peak_detected = any(abs(c - weak_c) < 1.5 for c in extracted_c)

    assert strong_peak_detected, "Algorithm failed to find the primary strong peak."

    # This will fail until the over-sparsification bug is fixed (e.g. by implementing
    # Orthogonal Matching Pursuit (OMP) or Joint Optimization)
    assert weak_peak_detected, (
        "Failure Mode Exposed: Algorithm over-subtracted the strong peak and completely "
        "destroyed/missed the overlapping weak peak in the tail."
    )
