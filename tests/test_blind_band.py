"""Blind-search quadrature and symmetry contracts, without a reference seed."""

import numpy as np
import pytest

from subhkl.search.blind_band import (
    count_search_weights,
    distinct_orientations,
    sample_signed,
)


def test_signed_quadrature_is_unbiased_and_keeps_negative_evidence():
    weights = np.array([-1.0, 3.0, 0.0])
    values = np.array([2.0, -1.0, 7.0])
    estimates = []
    negatives = 0
    for seed in range(3000):
        indices, sampled = sample_signed(weights, 2, seed)
        estimates.append(sampled @ values[indices])
        negatives += np.any(sampled < 0)
        assert 2 not in indices  # identically zero contribution
    assert negatives > 0
    expected = weights @ values / np.abs(weights).sum()
    assert abs(np.mean(estimates) - expected) < 0.03
    first = sample_signed(weights, 2, 77)
    second = sample_signed(weights, 2, 77)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_small_data_are_scored_exactly_including_negative_total():
    weights = np.array([-4.0, 1.0, -2.0])
    indices, sampled = sample_signed(weights, 10)
    np.testing.assert_array_equal(indices, np.arange(3))
    np.testing.assert_allclose(sampled, weights / 7)


@pytest.mark.parametrize("weights,size", [([0, 0], 4), ([1, np.nan], 4), ([1], 0)])
def test_invalid_search_evidence_is_rejected(weights, size):
    with pytest.raises(ValueError):
        sample_signed(weights, size)


def test_symmetry_equivalents_do_not_fill_shortlist():
    from scipy.spatial.transform import Rotation

    B = np.diag([1 / 70, 1 / 98, 1 / 91])
    U = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    equivalent = U @ np.diag([-1, 1, -1])
    other = Rotation.from_rotvec([0.5, -0.4, 0.2]).as_matrix()
    candidates = [(U, 3.0, 8.0), (equivalent, 2.9, 7.9), (other, 2.0, 5.0)]
    chosen = distinct_orientations(candidates, B, "P 1 21 1", 2)
    assert len(chosen) == 2
    np.testing.assert_allclose(chosen[0][0], U)
    np.testing.assert_allclose(chosen[1][0], other)


def test_signed_band_geometry_is_invariant_to_positive_weight_rescaling():
    import jax

    from subhkl.search.spherical import refine_band_geometry

    C = np.array([[0.2, 0.1, 0.2], [-0.2, 0.2, 0.3]])
    with jax.enable_x64():

        def fit(weights):
            return refine_band_geometry(
                C,
                np.zeros_like(C),
                weights,
                np.eye(3) / 12,
                np.eye(3),
                np.eye(3),
                (2.0, 4.0),
                2.0,
                n_zones=13,
                radial_bound=0.0,
                rot_bound_deg=0.0,
                trans_bound_m=0.0,
                orient_bound_deg=0.1,
                maxiter=1,
            )

        first = fit(np.array([-3.0, 1.0]))
        second = fit(np.array([-21.0, 7.0]))
    np.testing.assert_allclose(first["score"], second["score"], atol=1e-12)
    np.testing.assert_allclose(first["U"], second["U"], atol=1e-12)


def test_count_search_weights_are_threshold_free_and_suppress_null_noise():
    dev = np.array([0.0, 0.5, 1.0, 25.0, 100.0, 400.0])
    w = count_search_weights(dev)
    assert w[0] == 0.0
    assert np.all(w[1:] > 0.0)
    assert np.all(np.diff(w) > 0.0)
    # 1-sigma background fluctuation is suppressed by >1e4 relative to a 5-sigma spot
    assert w[2] / w[3] < 2e-4
    # Sublinear sqrt(D) growth at high deviance prevents flare monopolization
    assert np.isclose(w[5] / w[4], 2.0, rtol=1e-3)
    # Quantile cap prevents a single beamstop-flare bin from dominating
    flare = np.append(np.full(1000, 100.0), 1e8)
    w_capped = count_search_weights(flare, max_quantile=0.99, max_factor=3.0)
    assert np.isclose(w_capped[-1] / w_capped[0], 3.0, rtol=1e-3)


def test_refine_band_geometry_arbitrary_and_unconstrained_rot_axis():
    import jax

    from subhkl.search.spherical import refine_band_geometry

    C = np.array([[0.2, 0.1, 0.2], [-0.2, 0.2, 0.3], [0.1, -0.2, 0.25]])
    O = np.array([[0.01, -0.02, 0.0], [-0.01, 0.01, 0.02], [0.02, 0.0, -0.01]])
    w = np.array([2.0, 1.5, 3.0])
    with jax.enable_x64():
        fit_diag = refine_band_geometry(
            C,
            O,
            w,
            np.eye(3) / 12,
            np.eye(3),
            np.eye(3),
            (2.0, 4.0),
            2.0,
            n_zones=13,
            radial_bound=0.0,
            rot_bound_deg=1.0,
            rot_axis_only=(1.0, 1.0, 0.0),
            trans_bound_m=0.0,
            orient_bound_deg=0.0,
            maxiter=2,
        )
        rg_diag = fit_diag["global_rot_vec"]
        np.testing.assert_allclose(rg_diag[0], rg_diag[1], atol=1e-12)
        assert rg_diag[2] == 0.0

        fit_free = refine_band_geometry(
            C,
            O,
            w,
            np.eye(3) / 12,
            np.eye(3),
            np.eye(3),
            (2.0, 4.0),
            2.0,
            n_zones=13,
            radial_bound=0.0,
            rot_bound_deg=1.0,
            rot_axis_only=None,
            trans_bound_m=0.0,
            orient_bound_deg=0.0,
            maxiter=2,
        )
        assert fit_free["global_rot_vec"].shape == (3,)


def test_laue_order_stencil_and_band_envelope_recover_scissored_still():
    import jax
    from scipy.spatial.transform import Rotation

    from subhkl.core.crystallography import cartesian_matrix_metric_tensor
    from subhkl.search.spherical import (
        _quat_angle,
        direct_lattice_vectors,
        ghat_from_kf,
        lattice_ladder,
        laue_order_stencil,
        refine_band_geometry,
    )

    B, _ = cartesian_matrix_metric_tensor(35.0, 45.0, 40.0, 90.0, 102.0, 90.0)
    B = np.asarray(B, dtype=float)
    wl = (2.8, 4.5)
    d_min = 2.0
    st = laue_order_stencil(B, wl, d_min, orders=(-1, 0, 1))
    assert st.shape == (27, 3, 3)
    # Identity shift (0, 0, 0) is present and all matrices are proper rotations
    dets = np.linalg.det(st)
    np.testing.assert_allclose(dets, 1.0, atol=1e-12)
    assert any(np.allclose(R, np.eye(3), atol=1e-12) for R in st)

    # Synthetic Laue reflections under +5.5% radial / +4.2% axial detector scissor
    rng = np.random.default_rng(42)
    U_true = Rotation.from_rotvec([0.25, -0.35, 0.45]).as_matrix()
    hkls = rng.integers(-12, 13, size=(4000, 3))
    hkls = hkls[np.any(hkls != 0, axis=1)]
    g_cryst = (B @ hkls.T).T
    g_norm = np.linalg.norm(g_cryst, axis=1)
    hkls = hkls[(g_norm > 0.05) & (g_norm <= 1.0 / d_min)]
    g_lab = (U_true @ B @ hkls.T).T
    g2 = np.sum(g_lab**2, axis=1)
    # Ewald sphere condition: lambda = -2 * g_z / |g|^2
    lam = -2.0 * g_lab[:, 2] / np.maximum(g2, 1e-12)
    valid = (lam >= wl[0]) & (lam <= wl[1]) & (g_lab[:, 2] < -0.02)
    g_lab = g_lab[valid]
    lam = lam[valid]
    kf_true = g_lab + np.array([0.0, 0.0, 1.0]) / lam[:, None]
    kf_true /= np.linalg.norm(kf_true, axis=1, keepdims=True)
    # Project onto flat detector banks (nominal midpoints C_nom + within-panel offsets O_nom)
    xz_norm = np.hypot(kf_true[:, 0], kf_true[:, 2])
    valid_cyl = xz_norm > 0.25
    kf_true = kf_true[valid_cyl]
    scale_cyl = 0.40 / np.hypot(kf_true[:, 0], kf_true[:, 2])
    pos_true = kf_true * scale_cyl[:, None]
    s_rad_true, s_ax_true = 0.055, 0.042
    # Quantize panel midpoints to 8 azimuthal x 3 axial flat detector banks so
    # rigid-size panel offsets O_nom separate radial/axial scale from uniform zoom.
    phi = np.arctan2(pos_true[:, 0], pos_true[:, 2])
    phi_bank = np.round(phi / (np.pi / 4.0)) * (np.pi / 4.0)
    y_bank_true = np.round(np.clip(pos_true[:, 1], -0.25, 0.25) / 0.25) * 0.25
    mid_true = np.column_stack(
        [0.40 * np.sin(phi_bank), y_bank_true, 0.40 * np.cos(phi_bank)]
    )
    O_nom = pos_true - mid_true
    C_nom = np.column_stack(
        [
            mid_true[:, 0] / (1.0 + s_rad_true),
            mid_true[:, 1] / (1.0 + s_ax_true),
            mid_true[:, 2] / (1.0 + s_rad_true),
        ]
    )
    pos_nom = C_nom + O_nom
    kf_nom = pos_nom / np.linalg.norm(pos_nom, axis=1, keepdims=True)
    dirs_nom = ghat_from_kf(kf_nom)
    ki = np.array([0.0, 0.0, 1.0])
    sin_th_nom = np.linalg.norm(kf_nom - ki, axis=1) / 2.0
    weights = np.ones(len(C_nom), dtype=np.float64)

    with jax.enable_x64():
        found = lattice_ladder(
            dirs_nom,
            weights / weights.sum(),
            B,
            grid_deg=2.0,
            n_shortlist=200,
            coarse_zones=25,
            rungs=((25, 0.9, 1.0, 32, 4),),
            sin_theta=sin_th_nom,
            wavelength=wl,
            d_min=d_min,
            n_null=0,
            verbose=False,
            space_group="P21",
            band_envelope_deg=1.1,
            band_table_min_spacing_deg=0.30,
            table_tol_deg=0.8,
        )
        assert len(found) > 0
        cand_Us = np.stack([u for u, _, _ in found[:16]], axis=0)
        fit = refine_band_geometry(
            C_nom,
            O_nom,
            weights,
            B,
            np.eye(3),
            cand_Us,
            wl,
            d_min,
            n_zones=40,
            radial_bound=0.08,
            axial_bound=0.08,
            rot_bound_deg=1.0,
            trans_bound_m=0.0,
            orient_bound_deg=3.0,
            maxiter=60,
            space_group="P21",
            laue_order_search=True,
        )
    zd, _, _ = direct_lattice_vectors(B, space_group="P21")
    from subhkl.commands import _lattice_proper_rotations

    err_deg = min(
        np.degrees(_quat_angle(fit["U"], U_true @ S))
        for S in _lattice_proper_rotations(zd)
    )
    assert err_deg < 0.25
    assert abs(fit["radial_scale"] - s_rad_true) < 0.015
    assert abs(fit["axial_scale"] - s_ax_true) < 0.015
