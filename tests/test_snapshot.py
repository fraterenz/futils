import pytest
import numpy as np
from src.futils import snapshot


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
    excepted_x = np.array([[0, 1] * 2]).reshape(2, 2)

    result = snapshot.Uniformise.uniformise_histograms(histograms)
    assert result.x_max == expected_x_max
    assert np.testing.assert_array_equal(result.y, expected_y) is None
    assert np.testing.assert_array_equal(result.create_x_array(), excepted_x) is None


def test_uniformise():
    histograms = [snapshot.Histogram({10: 10, 11: 2}), snapshot.Histogram({0: 1, 1: 1})]
    expected_x_max = 11
    expected_y = np.zeros((2, expected_x_max + 1), dtype=int)
    expected_y[0, -1] = 2
    expected_y[0, -2] = 10
    expected_y[1, 0] = 1
    expected_y[1, 1] = 1
    excepted_x = np.array(list(range(0, 12)) * 2).reshape(2, 12)

    result = snapshot.Uniformise.uniformise_histograms(histograms)
    assert result.x_max == expected_x_max
    assert np.testing.assert_array_equal(result.y, expected_y) is None
    assert np.testing.assert_array_equal(result.create_x_array(), excepted_x) is None


def test_pooled_distribution_nothing_in_common():
    histograms = [snapshot.Histogram({10: 10, 11: 2}), snapshot.Histogram({0: 1, 1: 1})]
    distribution = snapshot.Uniformise.pooled_distribution(histograms)

    expected_keys = list(range(0, 12))
    expected_values = [0.0] * len(expected_keys)
    expected_values[0] = 1 / 14
    expected_values[1] = 1 / 14
    expected_values[-2] = 10 / 14
    expected_values[-1] = 2 / 14

    assert list(distribution.keys()) == expected_keys
    assert list(distribution.values()) == expected_values


def test_pooled_distribution_all_in_common():
    histograms = [
        snapshot.Histogram({10: 10, 11: 2}),
        snapshot.Histogram({10: 1, 11: 1}),
    ]
    distribution = snapshot.Uniformise.pooled_distribution(histograms)

    expected_keys = list(range(0, 12))
    expected_values = [0.0] * len(expected_keys)
    expected_values[-2] = 11 / 14
    expected_values[-1] = 3 / 14

    assert list(distribution.keys()) == expected_keys
    assert list(distribution.values()) == expected_values


def test_pooled_distribution_some_in_common():
    histograms = [
        snapshot.Histogram({10: 10, 11: 2, 4: 1}),
        snapshot.Histogram({0: 1, 1: 1, 4: 4}),
    ]
    distribution = snapshot.Uniformise.pooled_distribution(histograms)

    expected_keys = list(range(0, 12))
    expected_values = [0.0] * len(expected_keys)
    expected_values[0] = 1 / 19
    expected_values[1] = 1 / 19
    expected_values[4] = 5 / 19
    expected_values[-2] = 10 / 19
    expected_values[-1] = 2 / 19

    assert list(distribution.keys()) == expected_keys
    assert list(distribution.values()) == expected_values
