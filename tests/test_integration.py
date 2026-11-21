import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from subhkl.instrument.detector import Detector
from subhkl.integration import Peaks
from subhkl.optimization import FindUB

directory = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skip(
    reason="Integration test needs mesolite data and uses old API (harvest_peaks min_rel_intens param, opt.minimize method)"
)
def test_mesolite():
    # FIXME
    # Ignoring this test until we get a version of the mesolite input file constructed from the image files to test with

    directory = "tests/data"

    im_name = "meso_2_15min_2-0_4-5_050.tif"
    filename = os.path.join(directory, im_name)

    pks = Peaks(filename, "MANDI")
    xp, yp = pks.harvest_peaks(0, min_pix=30, min_rel_intens=0.05)

    ny, nx = pks.ims[0].shape

    r = 0.2
    p = 2 * np.pi * r * 180 / 180
    h = 0.45

    x, y = pks.scale_coordinates(xp, yp, p / nx, h / ny)
    fig, ax = plt.subplots(
        4, 1, figsize=(12.8, 19.2), sharex=False, layout="constrained"
    )

    extent = [-p / 2, p / 2, -h / 2, h / 2]

    ax[0].imshow(pks.ims[0], norm="log", cmap="binary", origin="lower", extent=extent)

    ax[0].minorticks_on()
    ax[0].set_aspect(1)

    ax[1].imshow(pks.ims[0], norm="log", cmap="binary", origin="lower", extent=extent)

    ax[1].scatter(x, y, edgecolor="r", facecolor="none")
    ax[1].minorticks_on()
    ax[1].set_aspect(1)
    detector = Detector(x, y, r, 0)
    two_theta, az_phi = detector.detector_trajectories()

    peaks_file = os.path.join(directory, "sucroae_imagine.h5")

    wl_min, wl_max = 2, 4.5

    with h5py.File(peaks_file, "w") as f:
        f["sample/a"] = 18.39
        f["sample/b"] = 56.55
        f["sample/c"] = 6.54
        f["sample/alpha"] = 90
        f["sample/beta"] = 90
        f["sample/gamma"] = 90
        f["sample/centering"] = "F"
        f["sample/B"] = np.diag([1 / 18.39, 1 / 56.55, 1 / 6.54])
        f["instrument/wavelength"] = [wl_min, wl_max]
        f["goniometer/R"] = np.eye(3)
        f["peaks/scattering"] = two_theta
        f["peaks/azimuthal"] = az_phi

    opt = FindUB(peaks_file)

    # Whether any run succeeded
    success = False

    # Number of attempted runs
    tries = 0

    while tries < 5:
        num, hkl, lamda = opt.minimize(48)

        ax[2].imshow(pks.im, norm="log", cmap="binary", origin="lower", extent=extent)

        ax[2].plot(x, y, "r.")
        ax[2].minorticks_on()
        ax[2].set_aspect(1)

        for i in range(len(hkl)):
            if np.linalg.norm(hkl[i]) > 0:
                coord = (x[i], y[i])
                label = "{:.0f}{:.0f}{:.0f}".format(*hkl[i])
                ax[2].annotate(label, coord)

        B = opt.reciprocal_lattice_B()
        U = opt.orientation_U(*opt.x)
        UB = opt.UB_matrix(U, B)

        Qx, Qy, Qz = np.einsum("ij,kj->ik", 2 * np.pi * UB, hkl)
        Q = np.sqrt(Qx**2 + Qy**2 + Qz**2)

        lamda = -4 * np.pi * Qz / Q**2
        mask = np.logical_and(lamda > wl_min, lamda < wl_max)

        Qx, Qy, Qz, Q, lamda = (
            Qx[mask],
            Qy[mask],
            Qz[mask],
            Q[mask],
            lamda[mask],
        )

        tt = -2 * np.arcsin(Qz / Q)
        az = np.arctan2(Qy, Qx)

        xv = np.sin(tt) * np.cos(az)
        yv = np.sin(tt) * np.sin(az)
        zv = np.cos(tt)

        xy_test_1 = np.allclose(xv**2 + yv**2 + zv**2, 1)

        t = r / np.sqrt(xv**2 + zv**2)

        xv *= t
        yv *= t
        zv *= t

        xy_test_2 = np.allclose(xv**2 + zv**2, r**2)

        # Proceed and stop retrying only if all the test conditions are true
        if num / len(lamda) > 0.5 and xy_test_1 and xy_test_2:
            success = True

            theta = np.arctan2(xv, zv)

            yp = yv.copy()
            xp = r * theta

            ax[3].imshow(
                pks.im, norm="log", cmap="binary", origin="lower", extent=extent
            )

            ax[3].scatter(x, y, edgecolor="r", facecolor="none")
            ax[3].plot(xp, yp, "w.")
            ax[3].minorticks_on()
            ax[3].set_aspect(1)

            name, _ext = os.path.splitext(im_name)

            directory = os.path.dirname(os.path.abspath(__file__))

            fig.savefig(os.path.join(directory, name + ".png"))

            break

    assert success


class _DummyMultiRunImage:
    def __init__(self, n_runs: int, bank_ids: list[int]):
        self.ims = {}
        self.bank_mapping = {}
        self._run_ids = {}
        idx = 0
        for r in range(n_runs):
            for b in bank_ids:
                self.ims[idx] = None
                self.bank_mapping[idx] = b
                self._run_ids[idx] = r
                idx += 1

    def get_run_id(self, img_key: int) -> int:
        return self._run_ids[img_key]


def test_predict_peaks_batch_gpu_matches_single_bank_flat_and_curved():
    """GPU batch predictor matches single-bank float64 kinematics and HKL order
    for both flat multi-panel (MANDI) and curved cylindrical (IMAGINE) detectors."""
    from scipy.spatial.transform import Rotation

    from subhkl.config import beamlines
    from subhkl.instrument.physics import predict_peaks_batch_gpu
    from subhkl.integration import orchestrator, worker

    a, b, c, alpha, beta, gamma = 32.0, 32.0, 96.0, 90.0, 90.0, 120.0
    fub = FindUB.__new__(FindUB)
    fub.a, fub.b, fub.c = a, b, c
    fub.alpha, fub.beta, fub.gamma = alpha, beta, gamma
    B_mat = fub.reciprocal_lattice_B()
    U_mat = Rotation.from_euler("xyz", [12.3, -34.5, 56.7], degrees=True).as_matrix()
    UB = U_mat @ B_mat

    gonio_axes = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    gonio_offsets = [0.5, -0.25, 1.0]
    harmonic_per_run = Rotation.from_euler(
        "xyz", [[0.0, 0.0, 0.0], [0.12, -0.06, 0.09]], degrees=True
    ).as_matrix()

    # 1. Flat-panel multi-run MANDI with goniometer, offsets, translations, harmonics
    mandi_banks = sorted(int(k) for k in beamlines["MANDI"] if str(k).isdigit())[:6]
    dummy_mandi = _DummyMultiRunImage(2, mandi_banks)
    f2r_mandi = np.repeat(np.arange(2), len(mandi_banks))
    tasks_mandi = orchestrator.prepare_predict_tasks(
        image_data=dummy_mandi,
        instrument="MANDI",
        wavelength_min=2.0,
        wavelength_max=4.0,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        d_min=1.3,
        UB=UB,
        space_group="P 61 2 2",
        sample_offset=[0.1, -0.1, 0.05],
        ki_vec=[0.0, 0.0, 1.0],
        gonio_axes=gonio_axes,
        gonio_angles=np.array([[0.0, 0.0, 0.0], [18.0, 5.0, -12.0]])[f2r_mandi],
        gonio_offsets=gonio_offsets,
        per_run_trans=np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]]),
        frame_to_run=f2r_mandi,
        harmonic_rot=harmonic_per_run[f2r_mandi],
    )
    res_mandi = predict_peaks_batch_gpu(tasks_mandi, show_progress=False)
    total_mandi = 0
    for t in tasks_mandi:
        _, (r_ref, c_ref, h_ref, k_ref, l_ref, wl_ref) = worker.predict_single_bank(*t)
        r_gpu, c_gpu, h_gpu, k_gpu, l_gpu, wl_gpu = res_mandi[t[0]]
        total_mandi += len(h_ref)
        np.testing.assert_array_equal(h_gpu, h_ref)
        np.testing.assert_array_equal(k_gpu, k_ref)
        np.testing.assert_array_equal(l_gpu, l_ref)
        np.testing.assert_allclose(r_gpu, r_ref, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(c_gpu, c_ref, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(wl_gpu, wl_ref, rtol=0.0, atol=1e-12)
    assert total_mandi > 500

    # 2. Curved cylindrical detector (IMAGINE) with non-zero 1D sample_offset
    img_banks = sorted(int(k) for k in beamlines["IMAGINE"] if str(k).isdigit())
    dummy_img = _DummyMultiRunImage(2, img_banks)
    f2r_img = np.repeat(np.arange(2), len(img_banks))
    tasks_img = orchestrator.prepare_predict_tasks(
        image_data=dummy_img,
        instrument="IMAGINE",
        wavelength_min=2.0,
        wavelength_max=4.5,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        d_min=1.6,
        UB=UB,
        space_group="P 61 2 2",
        sample_offset=[0.015, -0.005, 0.01],
        ki_vec=[0.0, 0.0, 1.0],
        gonio_axes=gonio_axes,
        gonio_angles=np.array([[0.0, 0.0, 0.0], [25.0, -4.0, 8.0]])[f2r_img],
        gonio_offsets=gonio_offsets,
        per_run_trans=np.array([[0.0, 0.0, 0.0], [0.01, -0.01, 0.02]]),
        frame_to_run=f2r_img,
        harmonic_rot=harmonic_per_run[f2r_img],
    )
    res_img = predict_peaks_batch_gpu(tasks_img, show_progress=False)
    total_img = 0
    for t in tasks_img:
        _, (r_ref, c_ref, h_ref, k_ref, l_ref, wl_ref) = worker.predict_single_bank(*t)
        r_gpu, c_gpu, h_gpu, k_gpu, l_gpu, wl_gpu = res_img[t[0]]
        total_img += len(h_ref)
        np.testing.assert_array_equal(h_gpu, h_ref)
        np.testing.assert_array_equal(k_gpu, k_ref)
        np.testing.assert_array_equal(l_gpu, l_ref)
        np.testing.assert_allclose(r_gpu, r_ref, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(c_gpu, c_ref, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(wl_gpu, wl_ref, rtol=0.0, atol=1e-12)
    assert total_img > 1000
