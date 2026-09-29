"""Repeated known-frontier experiments for comparing adaptive and fixed designs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from slamcart3d import __version__
from slamcart3d.core.provenance import source_sha256
from slamcart3d.experiments.statistics import summarize
from slamcart3d.robustness.benchmark import compare_sampling_methods, run_asrf_ablation


@dataclass(frozen=True)
class ValidationSuite:
    """Per-seed method comparisons and ablations for requested budgets."""

    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    comparisons: tuple[dict[str, float | int | str], ...]
    ablations: tuple[dict[str, float | int | str], ...]


def run_validation_suite(budgets: tuple[int, ...], seeds: tuple[int, ...]) -> ValidationSuite:
    """Repeat the known-boundary benchmark over budgets and random seeds."""

    if not budgets or any(budget < 8 for budget in budgets):
        raise ValueError("budgets must be non-empty and at least 8")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-empty and non-negative")
    comparisons: list[dict[str, float | int | str]] = []
    ablations: list[dict[str, float | int | str]] = []
    for budget in budgets:
        for seed in seeds:
            for row in compare_sampling_methods(budget, seed):
                comparisons.append({"requested_budget": budget, "seed": seed, **row})
    for seed in seeds:
        for row in run_asrf_ablation(max(budgets), seed):
            ablations.append({"requested_budget": max(budgets), "seed": seed, **row})
    return ValidationSuite(budgets, seeds, tuple(comparisons), tuple(ablations))


def _summarize_rows(
    rows: tuple[dict[str, float | int | str], ...], metrics: tuple[str, ...]
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((int(row["requested_budget"]), str(row["method"])), []).append(row)
    summaries: list[dict[str, float | int | str]] = []
    for (budget, method), group in sorted(groups.items()):
        summary: dict[str, float | int | str] = {
            "requested_budget": budget,
            "method": method,
            "replicates": len(group),
            "mean_simulations": sum(int(row["simulations"]) for row in group) / len(group),
        }
        for metric in metrics:
            statistics = summarize([float(row[metric]) for row in group])
            for key in ("mean", "standard_deviation", "ci95_low", "ci95_high"):
                summary[f"{metric}_{key}"] = float(statistics[key])
        summaries.append(summary)
    return summaries


def validation_summaries(
    suite: ValidationSuite,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Summarize comparison and ablation errors across design seeds."""

    metrics = ("boundary_classification_error", "fnr", "fpr")
    return _summarize_rows(suite.comparisons, metrics), _summarize_rows(suite.ablations, metrics)


def _write_dict_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("cannot export an empty validation table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_validation_suite(
    suite: ValidationSuite, output_root: str | Path = "results", plot: bool = False
) -> Path:
    """Export all observations, summaries, metadata, and optional curves."""

    created = datetime.now(timezone.utc).isoformat()
    digest = (
        sha256((created + repr(suite.budgets) + repr(suite.seeds)).encode("utf-8")).hexdigest()[:12].upper()
    )
    target = Path(output_root) / f"VALIDATION-{digest}"
    target.mkdir(parents=True, exist_ok=False)
    comparison_summary, ablation_summary = validation_summaries(suite)
    _write_dict_csv(target / "comparison_runs.csv", list(suite.comparisons))
    _write_dict_csv(target / "comparison_summary.csv", comparison_summary)
    _write_dict_csv(target / "ablation_runs.csv", list(suite.ablations))
    _write_dict_csv(target / "ablation_summary.csv", ablation_summary)
    metadata = {
        "validation_id": target.name,
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "budgets": suite.budgets,
        "seeds": suite.seeds,
        "true_boundary": "x + 0.35*sin(pi*y) + 0.30*y - 0.78 = 0",
        "validation_grid_resolution": 151,
        "ablation_budget": max(suite.budgets),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    if plot:
        from slamcart3d.visualization.plots import save_validation_curve

        save_validation_curve(comparison_summary, target / "boundary_error_vs_budget.png")
    return target
