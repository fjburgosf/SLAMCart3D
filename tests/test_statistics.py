import numpy as np

from slamcart3d.experiments.statistics import summarize


def test_summary_contains_requested_statistics() -> None:
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["iqr"] == 2.0
    assert np.isclose(result["standard_deviation"], np.sqrt(2.5))
