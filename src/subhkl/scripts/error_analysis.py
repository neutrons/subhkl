import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt


def analyze_errors(
    h5_file,
    threshold=0.2,
):
    # ----------------------------------------
    # Read HDF5
    # ----------------------------------------

    with h5py.File(h5_file, "r") as f:
        peak = f["metrics/per_peak"]

        h = peak["h"][:]
        k = peak["k"][:]
        l = peak["l"][:]

        d_err = peak["d_err"][:]
        ang_err = peak["ang_err"][:]

    print(f"Total peaks : {len(h)}")

    # ----------------------------------------
    # Plot ALL reflections
    # ----------------------------------------

    plt.figure(figsize=(18, 6))

    plt.scatter(
        np.arange(len(ang_err)),
        ang_err,
        s=20,
    )

    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"threshold={threshold}",
    )

    plt.xticks(fontsize=6)

    plt.xlabel("Peak index")
    plt.ylabel("Angular error (degree)")
    plt.title("Angular error for every reflection")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "ang_err_peak.png",
        dpi=300,
    )

    plt.close()

    print("Saved angular_error_allpeaks.png")

    # ----------------------------------------
    # High-error reflections
    # ----------------------------------------

    mask = ang_err > threshold

    high_h = h[mask]
    high_k = k[mask]
    high_l = l[mask]

    high_d = d_err[mask]
    high_ang = ang_err[mask]

    # sort by angular error

    order = np.argsort(high_ang)[::-1]

    high_h = high_h[order]
    high_k = high_k[order]
    high_l = high_l[order]

    high_d = high_d[order]
    high_ang = high_ang[order]

    # ----------------------------------------
    # Write CSV
    # ----------------------------------------

    with open(
        "high_error_hkl.csv",
        "w",
        newline="",
    ) as fout:
        writer = csv.writer(fout)

        writer.writerow(
            [
                "h",
                "k",
                "l",
                "d_err",
                "ang_err",
            ]
        )

        for row in zip(
            high_h,
            high_k,
            high_l,
            high_d,
            high_ang,
        ):
            writer.writerow(row)

    print(f"Saved high_error_hkl.csv ({len(high_h)} reflections)")

    # ----------------------------------------
    # Plot ONLY high-error reflections
    # ----------------------------------------

    high_labels = [
        f"({hh},{kk},{ll})"
        for hh, kk, ll in zip(
            high_h,
            high_k,
            high_l,
        )
    ]

    plt.figure(figsize=(16, 6))

    plt.bar(
        np.arange(len(high_ang)),
        high_ang,
    )

    plt.xticks(
        np.arange(len(high_labels)),
        high_labels,
        rotation=90,
        fontsize=7,
    )

    plt.ylabel("Angular error (degree)")
    plt.xlabel("High-error HKLs")

    plt.title(f"Reflections with angular error > {threshold}")

    plt.tight_layout()

    plt.savefig(
        "high_error_hkl.png",
        dpi=300,
    )

    plt.close()

    print("Saved high_error_hkl.png")

    return {
        "num_peaks": len(h),
        "high_error": len(high_h),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("h5_file")

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    analyze_errors(
        args.h5_file,
        threshold=args.threshold,
    )
