import numpy as np
from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

# =====================================================================
# MOCK OBJECTS FOR INTEGRATOR PIPELINE
# =====================================================================


class MockDetector:
    def __init__(self):
        self.uhat = np.array([1.0, 0.0, 0.0])  # X-axis
        self.vhat = np.array([0.0, 1.0, 0.0])  # Y-axis
        self.width = 0.150  # 150 mm in METERS
        self.height = 0.150  # 150 mm in METERS
        self.m = 150
        self.n = 150

    def pixel_to_angles(self, rs, cs, sample_offset=None, ki_vec=None):
        return np.zeros_like(rs), np.zeros_like(cs)

    def pixel_to_lab(self, r, c):
        r = np.atleast_1d(r)
        c = np.atleast_1d(c)
        xyz = np.zeros((len(r), 3))

        # 1 pixel = 1 mm = 0.001 m
        xyz[:, 0] = (c - 75.0) * 0.001
        xyz[:, 1] = (r - 75.0) * 0.001
        xyz[:, 2] = 0.100  # 100mm detector distance in METERS

        return xyz if len(xyz) > 1 else xyz[0]


class MockImage:
    def __init__(self, ims):
        self.ims = ims
        self.bank_mapping = {k: "bank1" for k in ims.keys()}

    def get_run_id(self, img_key):
        return int(img_key.split("_")[1])


class MockPeaksObj:
    def __init__(self, ims):
        self.image = MockImage(ims)
        self.instrument = "MANDI"

    def get_detector_by_img(self, img_key):
        return MockDetector()

    def get_run_id(self, img_key):
        return self.image.get_run_id(img_key)

    def get_image_label(self, img_key):
        return img_key


# =====================================================================
# SYNTHETIC DATA GENERATOR
# =====================================================================


def generate_synthetic_data():
    ims = {}
    peak_dict = {}

    # Generate 20 peaks (must be >= 15 to trigger 3D global shape optimization)
    rs = np.linspace(30, 120, 4)
    cs = np.linspace(30, 120, 5)
    rr, cc = np.meshgrid(rs, cs)
    r_flat = rr.flatten()
    c_flat = cc.flatten()

    num_peaks = len(r_flat)
    h_arr = np.arange(1, num_peaks + 1)
    k_arr = np.ones(num_peaks)
    l_arr = np.ones(num_peaks)
    wl_arr = np.ones(num_peaks)

    for run_id in [0, 1]:
        img_key = f"img_{run_id}"
        img = np.ones((150, 150), dtype=np.float32) * 5.0  # Flat noise background

        yy, xx = np.indices((150, 150))
        for i in range(num_peaks):
            r_c, c_c = r_flat[i], c_flat[i]

            # Simulate a 3D crystal that is physically stretched.
            # Run 0 (0 degrees): Appears horizontally stretched on detector.
            # Run 1 (90 degrees around Z): Appears vertically stretched on detector.
            if run_id == 0:
                var_r, var_c = 1.0, 9.0
            else:
                var_r, var_c = 9.0, 1.0

            gaussian = 100.0 * np.exp(
                -0.5 * ((yy - r_c) ** 2 / var_r + (xx - c_c) ** 2 / var_c)
            )
            img += gaussian

        ims[img_key] = img
        peak_dict[img_key] = (r_flat, c_flat, h_arr, k_arr, l_arr, wl_arr)

    return ims, peak_dict


# =====================================================================
# THE TEST
# =====================================================================


def test_rbf_integrator_goniometer_kinematics():
    ims, peak_dict = generate_synthetic_data()
    peaks_obj = MockPeaksObj(ims)

    # Mock a rotation around the Z-axis (beam axis)
    gonio_axes = np.array([[0.0, 0.0, 1.0, 1.0]])

    # Run 0 is at 0 degrees, Run 1 is rotated by 90 degrees
    gonio_angles = np.array([[0.0], [90.0]])
    gonio_offsets = np.array([0.0])

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=peaks_obj,
        sigmas=[1.0, 2.0, 3.0],
        show_progress=False,
        nominal_sigma=2.0,
        anisotropic=True,
        fit_mosaicity=False,
        border_width=5,
        chunk_size=128,
        gonio_axes=gonio_axes,
        gonio_angles=gonio_angles,
        gonio_offsets=gonio_offsets,
    )

    # 1. Assert all peaks were successfully integrated
    assert len(res.intensity) == 40, f"Expected 40 peaks, got {len(res.intensity)}"

    # 2. Assert OLS Collinearity Bug is resolved (intensities strictly positive)
    intensities = np.array(res.intensity)
    assert np.all(intensities > 0), (
        "Found negative intensities, the unconstrained OLS solver failed."
    )

    # 3. Split integration metrics by Run ID
    run_0_mask = np.array(res.run_id) == 0
    run_1_mask = np.array(res.run_id) == 1

    var_u_0 = np.array(res.var_u)[run_0_mask]  # Horizontal variance
    var_v_0 = np.array(res.var_v)[run_0_mask]  # Vertical variance

    var_u_1 = np.array(res.var_u)[run_1_mask]
    var_v_1 = np.array(res.var_v)[run_1_mask]

    # 4. Assert Kinematic Projection Accuracy
    # Run 0 should be stretched horizontally
    assert np.mean(var_u_0) > np.mean(var_v_0), (
        "Run 0 did not recover horizontal major axis."
    )

    # Run 1 should be stretched vertically due to the 90-degree goniometer rotation.
    # If the integrator failed to rotate the 3D tensor, this assertion will fail.
    assert np.mean(var_v_1) > np.mean(var_u_1), (
        "Run 1 did not correctly project the rotated 3D tensor."
    )
