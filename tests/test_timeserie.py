import pytest
import numpy as np
from futils import timeserie


def test_empty_histogram():
    with pytest.raises(AssertionError, match=r"list of histograms is empty"):
        timeserie.Uniformise.make_array([])


def test_one_histogram():
    dynamics = [timeserie.Timeserie([1, 1, 2, 1])]
    with pytest.raises(AssertionError, match=r"found only one histogram"):
        timeserie.Uniformise.make_array(dynamics)


def test_uniformise_already_uniformised():
    dynamics = [timeserie.Timeserie([1, 1, 2, 1]), timeserie.Timeserie([0, 0, 0, 1])]
    expected = np.array([[1, 1, 2, 1], [0, 0, 0, 1]], dtype=float)

    result = timeserie.Uniformise.make_array(dynamics)
    assert np.testing.assert_array_equal(result, expected) is None


def test_uniformise():
    dynamics = [timeserie.Timeserie([1, 1, 2]), timeserie.Timeserie([0, 0, 0, 1])]
    expected = np.array([[1, 1, 2, np.nan], [0, 0, 0, 1]], dtype=float)

    result = timeserie.Uniformise.make_array(dynamics)
    assert np.testing.assert_array_equal(result, expected) is None
