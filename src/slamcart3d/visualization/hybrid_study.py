"""Comparative figure for the fixed-condition local SLAM benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from slamcart3d.visualization.plots import _prepare_plot_cache

if TYPE_CHECKING:
    from slamcart3d.experiments.hybrid_validation import HybridBenchmark


def save_hybrid_ate_figure(benchmark: HybridBenchmark, path: str | Path) -> Path:
    """Plot seed-level ATE means with descriptive 95% intervals."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    count = len(benchmark.scenarios)
    figure, axes = plt.subplots(
        count, 1, figsize=(9, max(3.2 * count, 4.0)), squeeze=False, constrained_layout=True
    )
    for axis, scenario in zip(axes[:, 0], benchmark.scenarios, strict=True):
        rows = [
            row
            for row in benchmark.summaries
            if row["scenario"] == scenario.name and row["metric"] == "ate_rmse_m"
        ]
        for index, row in enumerate(rows):
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            color = "#0b7a53" if row["method"] == "adaptive_hybrid" else "#52637b"
            axis.errorbar(
                mean,
                index,
                xerr=[[max(0.0, mean - low)], [max(0.0, high - mean)]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
            )
        axis.set_yticks(range(len(rows)), [str(row["method"]).replace("_", " ") for row in rows])
        axis.invert_yaxis()
        axis.set_title(scenario.name.replace("_", " "))
        axis.set_xlabel("ATE RMSE [m]; lower is better")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle(f"Local SLAM comparison (n={len(benchmark.seeds)} paired seeds)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination
