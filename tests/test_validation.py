import csv

from slamcart3d.robustness.validation import export_validation_suite, run_validation_suite


def test_repeated_validation_exports_all_methods_and_seeds(tmp_path) -> None:
    suite = run_validation_suite((16,), (0, 1))
    assert len(suite.comparisons) == 8
    assert len(suite.ablations) == 6
    target = export_validation_suite(suite, tmp_path)
    with (target / "comparison_runs.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 8
    assert {row["method"] for row in rows} == {"grid", "random", "lhs", "asrf"}
    assert (target / "ablation_summary.csv").exists()
