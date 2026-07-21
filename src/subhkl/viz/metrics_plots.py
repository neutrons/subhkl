import numpy as np
import matplotlib.pyplot as plt


def plot_error_histograms(result: dict, out_name: str = "metrics_histograms.png") -> None:
    """Plot overall d_err/ang_err distributions from a
    compute_metrics(..., return_per_peak=True) result."""
    per_peak = result.get("per_peak")
    if per_peak is None:
        raise ValueError(
            "result has no 'per_peak' data; call compute_metrics(..., return_per_peak=True)"
        )

    d_err = per_peak["d_err"]
    ang_err = per_peak["ang_err"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(d_err, bins=50, color="tab:blue", edgecolor="black")
    axes[0].set_xlabel("d spacing error")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"d_err (n={len(d_err)})")

    axes[1].hist(ang_err, bins=50, color="tab:orange", edgecolor="black")
    axes[1].set_xlabel("angular error (deg)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"ang_err (n={len(ang_err)})")

    fig.tight_layout()
    fig.savefig(out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_run_histograms(
    result: dict, out_name: str = "metrics_histograms_per_run.png"
) -> None:
    """Plot one row of d_err/ang_err histograms per run/frame from a
    compute_metrics(..., return_per_peak=True) result."""
    per_peak = result.get("per_peak")
    if per_peak is None:
        raise ValueError(
            "result has no 'per_peak' data; call compute_metrics(..., return_per_peak=True)"
        )

    run_index = np.asarray(per_peak["run_index"])
    d_err = np.asarray(per_peak["d_err"])
    ang_err = np.asarray(per_peak["ang_err"])

    runs = sorted(np.unique(run_index))
    n_runs = len(runs)

    d_bins = np.histogram_bin_edges(d_err, bins=30)
    ang_bins = np.histogram_bin_edges(ang_err, bins=30)

    fig, axes = plt.subplots(
        n_runs, 2, figsize=(10, max(2.2 * n_runs, 3)), squeeze=False
    )

    for row, run in enumerate(runs):
        mask = run_index == run
        ax_d, ax_ang = axes[row]

        ax_d.hist(d_err[mask], bins=d_bins, color="tab:blue", edgecolor="black")
        ax_ang.hist(ang_err[mask], bins=ang_bins, color="tab:orange", edgecolor="black")

        ax_d.set_ylabel(f"run {run}\ncount")
        if row == 0:
            ax_d.set_title("d_err")
            ax_ang.set_title("ang_err")
        if row == n_runs - 1:
            ax_d.set_xlabel("d spacing error")
            ax_ang.set_xlabel("angular error (deg)")

    fig.tight_layout()
    fig.savefig(out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
