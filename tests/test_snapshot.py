import pytest
import numpy as np
from futils import snapshot


def test_empty_histogram():
    with pytest.raises(AssertionError, match=r"list of histograms is empty"):
        snapshot.Uniformise.uniformise_histograms([])


def test_one_histogram():
    histograms = [snapshot.Histogram({0: 10, 1: 2})]
    with pytest.raises(AssertionError, match=r"found only one histogram"):
        snapshot.Uniformise.uniformise_histograms(histograms)


def test_uniformise_already_uniformised():
    histograms = [snapshot.Histogram({0: 10, 1: 2}), snapshot.Histogram({0: 1, 1: 1})]
    expected_x_max = 1
    expected_y = np.array([[10, 2], [1, 1]], dtype=int)

    result = snapshot.Uniformise.uniformise_histograms(histograms)
    assert result.x_max == expected_x_max
    assert np.testing.assert_array_equal(result.y, expected_y) is None


def test_uniformise():
    histograms = [snapshot.Histogram({10: 10, 11: 2}), snapshot.Histogram({0: 1, 1: 1})]
    expected_x_max = 11
    expected_y = np.zeros((2, expected_x_max + 1), dtype=int)
    expected_y[0, -1] = 2
    expected_y[0, -2] = 10
    expected_y[1, 0] = 1
    expected_y[1, 1] = 1

    result = snapshot.Uniformise.uniformise_histograms(histograms)
    assert result.x_max == expected_x_max
    assert np.testing.assert_array_equal(result.y, expected_y) is None
