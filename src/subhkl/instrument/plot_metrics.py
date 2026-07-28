import h5py
import matplotlib.pyplot as plt


def plot_metric(file1, metric="median_ang_err", output="metric.png"):

    with h5py.File(file1, "r") as f:
        table = f["metrics/per_hkl"][:]

    if metric not in table.dtype.names:
        raise ValueError(f"{metric} not found")

    plt.figure(figsize=(8,5))

    plt.scatter(
        range(len(table)),
        table[metric],
        s=20,
    )

    plt.xlabel("Unique HKL")
    plt.ylabel(metric)

    plt.tight_layout()

    plt.savefig(output, dpi=300)

    print(f"Saved {output}")
