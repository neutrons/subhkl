import sys

import gemmi
import h5py
import numpy as np


class BaseConcatenateMerger:
    def __init__(self, h5_files, copy_keys, merge_keys, per_file_keys=None):
        """
        Merges datasets by concatenation

        Parameters
        ----------
        h5_files : list[str]
            List of .h5 file paths
        copy_keys : list[str]
            List of keys in .h5 files to copy once
        merge_keys : list[str]
            List of keys in .h5 files to merge by concatenation
        per_file_keys : list[str]
            List of keys that are per-run/per-file (shape [N_runs, ...])
        """
        self.h5_files = sorted(list(set(h5_files)))
        self.copy_keys = copy_keys
        self.merge_keys = merge_keys
        self.per_file_keys = per_file_keys if per_file_keys is not None else []

    def merge(self, output_filename):
        """
        Merges the indexed datasets into a single dataset

        Parameters
        ----------
        output_filename : str
            Name of the output .h5 file to write the merged dataset to
        """

        total_peaks = 0
        total_runs = 0
        # Determine total peaks AND find a valid template file
        typical_file_path = self.h5_files[0]
        found_valid_template = False

        for file in self.h5_files:
            with h5py.File(file, "r") as f_in:
                num = len(f_in[self.merge_keys[0]])
                total_peaks += num

                num_runs_in_file = 0
                if self.per_file_keys:
                    # Assume each file is ONE run for now, OR check length of first per_file_key
                    if self.per_file_keys[0] in f_in:
                        num_runs_in_file = len(f_in[self.per_file_keys[0]])

                if num_runs_in_file == 0:
                    if "peaks/run_index" in f_in and len(f_in["peaks/run_index"]) > 0:
                        num_runs_in_file = int(np.max(f_in["peaks/run_index"]) + 1)
                    else:
                        num_runs_in_file = 1

                total_runs += num_runs_in_file

                # Use the first file with data as the template to ensure
                # multidimensional shapes (like 3x3 matrices) are preserved.
                if not found_valid_template and num > 0:
                    typical_file_path = file
                    found_valid_template = True

        with h5py.File(output_filename, "w") as f_out:
            # Open the valid template file we found (or the first one if all are empty)
            with h5py.File(typical_file_path, "r") as f_typical:
                # Root attributes ride along (the reduced files record e.g.
                # instrument='CG4D' there).  Dropping them was how a merged
                # file ended up unable to say which instrument it belongs
                # to, so every replay command downstream needed the caller
                # to repeat what the inputs already knew.
                for name, value in f_typical.attrs.items():
                    f_out.attrs[name] = value

                for key in self.copy_keys:
                    if key in f_typical:
                        f_out[key] = np.array(f_typical[key])

                for merge_key in self.merge_keys:
                    # Use the shape from the typical file
                    shape = (total_peaks,) + f_typical[merge_key].shape[1:]
                    dtype = f_typical[merge_key].dtype
                    f_out.create_dataset(merge_key, shape, dtype)

                for per_file_key in self.per_file_keys:
                    if per_file_key in f_typical:
                        shape = (total_runs,) + f_typical[per_file_key].shape[1:]
                        dtype = f_typical[per_file_key].dtype
                        f_out.create_dataset(per_file_key, shape, dtype)

            offset = 0
            run_offset = 0
            f_out["files"] = np.array(
                list(map(lambda s: s.encode("utf-8"), self.h5_files))
            )
            f_out.create_dataset("file_offsets", (len(self.h5_files),), dtype=np.int64)

            for i_file, indexed_file in enumerate(self.h5_files):
                with h5py.File(indexed_file, "r") as f_in:
                    num_items = len(f_in[self.merge_keys[0]])
                    peak_range = slice(offset, offset + num_items)
                    f_out["file_offsets"][i_file] = offset

                    # 1. Merge per-peak keys
                    for merge_key in self.merge_keys:
                        if merge_key in f_in:
                            # Only copy if there is data to avoid shape mismatch on empty files
                            if num_items > 0:
                                data = np.array(f_in[merge_key])
                                # Increment run_index or image_index if it's per-file local
                                if (
                                    merge_key == "peaks/run_index"
                                    or merge_key == "peaks/image_index"
                                ):
                                    # We assign a global run index based on the running offset.
                                    # This ensures indices from multiple files do not collide.
                                    data += run_offset

                                f_out[merge_key][peak_range] = data
                        elif num_items > 0 and merge_key == "peaks/run_index":
                            # Fallback: if no run_index exists in input, we assign global run offset
                            f_out["peaks/run_index"][peak_range] = run_offset

                    # 2. Merge per-file/run keys
                    num_runs_in_file = 0
                    if self.per_file_keys:
                        for per_file_key in self.per_file_keys:
                            if per_file_key in f_in:
                                data = np.array(f_in[per_file_key])
                                n_r = len(data)
                                num_runs_in_file = max(num_runs_in_file, n_r)
                                run_range = slice(run_offset, run_offset + n_r)
                                f_out[per_file_key][run_range] = data

                    if num_runs_in_file == 0 and num_items > 0:
                        if "peaks/run_index" in f_in:
                            num_runs_in_file = int(np.max(f_in["peaks/run_index"]) + 1)
                        else:
                            num_runs_in_file = 1

                    offset += num_items
                    run_offset += num_runs_in_file


class FinderConcatenateMerger(BaseConcatenateMerger):
    def __init__(self, h5_files):
        merge_keys = [
            "wavelength_mins",
            "wavelength_maxes",
            "peaks/two_theta",
            "peaks/azimuthal",
            "peaks/intensity",
            "peaks/sigma",
            "peaks/radius",  # Added radius to merge keys
            "peaks/xyz",
            "bank",
            "peaks/image_index",
            "peaks/run_index",
            "goniometer/R",
            "goniometer/angles",
        ]
        per_file_keys = []
        copy_keys = ["goniometer/axes", "goniometer/names"]
        super().__init__(h5_files, copy_keys, merge_keys, per_file_keys=per_file_keys)


class MTZExporter:
    """Unmerged MTZ from an integrator output.

    With ``predictions_file`` / ``corrections_file``, per-observation
    systematics proxies ride along as extra real-valued columns so a
    downstream scaling model (careless) can LEARN the residual error
    surface instead of having it folded silently into the merged
    intensities:

    - SIGEFF [px]: the projected peak radius (det Sigma)^(1/4);
    - SNAPD [px]: distance from the integrated position to the nearest
      predicted position on the same image -- how far the geometry was
      off for THIS peak, the direct flux-error proxy;
    - DPHI [deg], DTX/DTY/DTZ [mm]: the fitted per-run goniometer angle
      and translation corrections of the peak's run.  As per-run
      constants these are spanned by image-wise scales; their value to
      the scale model is in interactions with the per-peak geometry
      (a run whose phi was 0.7 deg off does not err uniformly).

    Columns appear only when their source file is supplied, so existing
    exports are byte-identical.
    """

    def __init__(
        self,
        peaks_file,
        space_group=None,
        predictions_file=None,
        corrections_file=None,
    ):
        with h5py.File(peaks_file) as f:
            self.a = float(np.array(f["sample/a"]))
            self.b = float(np.array(f["sample/b"]))
            self.c = float(np.array(f["sample/c"]))
            self.alpha = float(np.array(f["sample/alpha"]))
            self.beta = float(np.array(f["sample/beta"]))
            self.gamma = float(np.array(f["sample/gamma"]))

            if space_group is None:
                sg = f["sample/space_group"][()]
                if isinstance(sg, bytes):
                    space_group = sg.decode("utf-8")
                else:
                    space_group = sg

            self.h = np.array(f["peaks/h"])
            self.k = np.array(f["peaks/k"])
            self.l = np.array(f["peaks/l"])
            self.lamda = np.array(f["peaks/lambda"])
            self.theta = np.array(f["peaks/two_theta"]) / 2
            self.phi = np.array(f["peaks/azimuthal"])
            self.intensity = np.array(f["peaks/intensity"])
            self.sigma = np.array(f["peaks/sigma"])
            if "structure_factors" in f["peaks"].keys():
                self.f = np.array(f["peaks/structure_factors"])
                self.f_sigma = np.array(f["peaks/structure_factors_sigma"])
            else:
                self.f = None
                self.f_sigma = None

            if "run_index" in f["peaks"].keys():
                self.runs = np.array(f["peaks/run_index"])
            else:
                self.runs = np.zeros_like(self.h, dtype=np.int32)

            run_raw = self.runs.copy()
            self.runs = 1000 * self.runs + f["peaks/bank"]

            self.extra = {}  # column name -> (mtz type, per-row values)
            has_shape = all(k in f["peaks"] for k in ("var_u", "var_v", "cov_uv"))
            if (predictions_file or corrections_file) and has_shape:
                var_u = np.array(f["peaks/var_u"], dtype=float)
                var_v = np.array(f["peaks/var_v"], dtype=float)
                cov_uv = np.array(f["peaks/cov_uv"], dtype=float)
                det = np.maximum(var_u * var_v - cov_uv**2, 1e-6)
                self.extra["SIGEFF"] = ("R", det**0.25)
            if predictions_file is not None:
                img = np.array(f["peaks/image_index"], dtype=int)
                pr = np.array(f["peaks/pixel_r"], dtype=float)
                pc = np.array(f["peaks/pixel_c"], dtype=float)
                snapd = np.zeros(len(pr))
                with h5py.File(predictions_file) as fp:
                    pred = {
                        int(k): np.stack(
                            [fp[f"banks/{k}/i"][()], fp[f"banks/{k}/j"][()]],
                            axis=1,
                        )
                        for k in fp["banks"]
                    }
                for i in range(len(pr)):
                    P = pred.get(img[i])
                    if P is None or len(P) == 0:
                        continue
                    d2 = (P[:, 0] - pr[i]) ** 2 + (P[:, 1] - pc[i]) ** 2
                    # Cap at 5 px: a farther "match" is a bookkeeping
                    # mismatch, not a measured displacement.
                    snapd[i] = min(float(np.sqrt(d2.min())), 5.0)
                self.extra["SNAPD"] = ("R", snapd)
            if corrections_file is not None:
                with h5py.File(corrections_file) as fc:
                    g = fc.get("goniometer/per_run")
                    if g is not None:
                        n_runs = len(g["delta_deg"])
                        run_c = np.clip(run_raw, 0, n_runs - 1)
                        if "delta_deg" in g:
                            self.extra["DPHI"] = (
                                "R",
                                np.array(g["delta_deg"])[run_c],
                            )
                        if "trans_m" in g:
                            t_mm = 1e3 * np.array(g["trans_m"])[run_c]
                            for j, ax in enumerate(("DTX", "DTY", "DTZ")):
                                self.extra[ax] = ("R", t_mm[:, j])

        self.space_group = space_group

    def write_mtz(self, filename):
        mtz = gemmi.Mtz(with_base=True)
        mtz.set_logging(sys.stdout)

        sg = gemmi.find_spacegroup_by_name(self.space_group)
        if sg is None:
            raise ValueError(f"Could not find space group: {self.space_group}")
        mtz.spacegroup = sg

        unit_cell = gemmi.UnitCell(
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma
        )
        mtz.set_cell_for_all(unit_cell)

        mtz.add_column("I", "J")
        mtz.add_column("SIGI", "Q")
        if self.f is not None:
            mtz.add_column("FP", "F")
            mtz.add_column("SIGFP", "Q")
        mtz.add_column("WAVEL", "W")
        mtz.add_column("THETA", "W")
        mtz.add_column("PHI", "W")
        mtz.add_column("BATCH", "B")
        for name, (mtz_type, _vals) in self.extra.items():
            mtz.add_column(name, mtz_type)

        # Column order: h, k, l, I, sigI, [FP, sigFP,] wavel, theta, phi,
        # batch, [extras...]
        n_base_cols = 9  # h, k, l, I, sigI, wavel, theta, phi, batch
        n_structure_factor_cols = 2  # FP, sigFP
        n_cols = (
            n_base_cols
            + (n_structure_factor_cols if self.f is not None else 0)
            + len(self.extra)
        )

        data = []

        for i in range(len(self.intensity)):
            h, k, l = self.h[i], self.k[i], self.l[i]  # noqa: E741

            # Drop invalid peaks
            if h == 0 and k == 0 and l == 0:
                continue

            # A nonpositive sigma is a marker, not a measurement: the
            # matrix-free integrator stamps sigI = 0 on amplitudes pinned
            # at its nonnegativity boundary (censored observations that
            # would bias weak merged intensities if exported as 0 +- sigma).
            if self.sigma[i] <= 0.0:
                continue

            intensity, sigma = self.intensity[i], self.sigma[i]
            wl = self.lamda[i]
            theta = self.theta[i]
            phi = self.phi[i]

            if self.runs is not None:
                run = self.runs[i]
            else:
                run = 0

            if self.f is not None:
                f, f_sigma = self.f[i], self.f_sigma[i]
                row = [
                    h,
                    k,
                    l,
                    intensity,
                    sigma,
                    f,
                    f_sigma,
                    wl,
                    theta,
                    phi,
                    run,
                ]
            else:
                row = [h, k, l, intensity, sigma, wl, theta, phi, run]
            row.extend(float(vals[i]) for _t, vals in self.extra.values())

            data.append(row)

        if len(data) == 0:
            # Empty case: create a 2D array with correct number of columns
            data = np.empty((0, n_cols), dtype=np.float32)
        else:
            data = np.ascontiguousarray(np.array(data, dtype=np.float32))

        mtz.set_data(data)
        mtz.write_to_file(filename)


class ImageStackMerger(BaseConcatenateMerger):
    def __init__(self, h5_files):
        """
        Merges reduced image HDF5 files into a single stack for batch processing.
        """
        merge_keys = [
            "images",  # The stack of 2D images
            "goniometer/angles",  # Per-image angles
            "bank_ids",  # Per-image detector ID
        ]

        # Keys that should be identical across all files (metadata)
        copy_keys = [
            "goniometer/axes",
            "goniometer/names",
            "instrument/wavelength",
            "instrument/name",
        ]
        super().__init__(h5_files, copy_keys, merge_keys)
