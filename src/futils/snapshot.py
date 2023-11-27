"""We load and process a histogram representing a single snapshot of a
simulation at a certain timepoint.
The histogram are stored as json files with keys being a quantity of interest
and the values are the number of individuals.
"""
import numpy as np
import json
from typing import Dict, List, NewType, Tuple
from pathlib import Path


Distribution = NewType("Distribution", Dict[int, float])
Histogram = NewType("Histogram", Dict[int, int])


def histogram_from_dict(my_dict: Dict[int, int]) -> Histogram:
    sorted_dict = dict(sorted(my_dict.items()))
    return Histogram({int(x): int(y) for x, y in sorted_dict.items()})


def histogram_from_file(file: Path) -> Histogram:
    with open(file, "r") as f:
        return histogram_from_dict(json.load(f))


def cdf_from_histogram(hist: Histogram) -> Tuple[np.ndarray, np.ndarray]:
    """
    >>> from src.futils import snapshot
    >>> my_keys = [1, 0, 3]
    >>> my_values = [2, 2, 1]
    >>> histogram = snapshot.Histogram({k: ele for k, ele in zip(my_keys, my_values)})
    >>> cdf = snapshot.cdf_from_histogram(histogram)
    >>> cdf[0]
    array([0, 1, 3])
    >>> cdf[1]
    array([0.4, 0.8, 1. ])
    """
    distr = distribution_from_histogram(hist)
    return cdf_from_distribution(distr)


def cdf_from_distribution(distr: Distribution) -> Tuple[np.ndarray, np.ndarray]:
    """Return the x-axis (all the possible states) and the cumulative
    distribution (y-axis)
    >>> from src.futils import snapshot
    >>> my_keys = [1, 0, 3]
    >>> my_values = [2, 2, 1]
    >>> distr = snapshot.Distribution({k: ele/5 for k, ele in zip(my_keys, my_values)})
    >>> cdf = snapshot.cdf_from_distribution(distr)
    >>> cdf[0]
    array([0, 1, 3])
    >>> cdf[1]
    array([0.4, 0.8, 1. ])
    """
    ordered_distr = dict(sorted(distr.items()))
    return np.array(list(ordered_distr.keys())), np.array(
        np.cumsum(list(ordered_distr.values()), dtype=float), dtype=float
    )


def distribution_from_histogram(hist: Histogram) -> Distribution:
    """
    >>> from src.futils import snapshot
    >>> my_keys = [0, 1, 3]
    >>> my_values = [2, 2, 1]
    >>> histogram = snapshot.Histogram({k: ele for k, ele in zip(my_keys, my_values)})
    >>> distribution = snapshot.distribution_from_histogram(histogram)
    >>> assert list(distribution.keys()) == my_keys
    >>> list(distribution.keys())
    [0, 1, 3]
    >>> [round(ele, 2) for ele in distribution.values()]
    [0.4, 0.4, 0.2]
    """
    ntot = sum(hist.values())
    return Distribution({k: ele / ntot for k, ele in hist.items()})


class HistogramsUniformised:
    x_max: int
    y: np.ndarray  # shape: nb_of_histograms x x_max

    def __init__(self, x_max: int, y: np.ndarray) -> None:
        assert x_max == y.shape[1] - 1, f"{x_max} vs {y.shape[1] - 1}"
        self.x_max = x_max
        self.y = y

    def make_histograms(self) -> List[Histogram]:
        """
        >>> from src.futils import snapshot
        >>> import numpy as np
        >>> my_values = np.array([[1, 0, 3, 1], [0, 0, 1, 1]], dtype=int)
        >>> uniformised_hists = snapshot.HistogramsUniformised(x_max=3, y=my_values)
        >>> uniformised_hists.make_histograms()
        [{0: 1, 1: 0, 2: 3, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}]
        """
        histograms = []
        for idx in range(self.get_nb_histograms()):
            histograms.append(
                {k: val for k, val in zip(self.create_x_keys(), self.y[idx, :])}
            )
        return histograms

    def get_nb_histograms(self) -> int:
        return self.y.shape[0]

    def create_x_keys(self) -> List[int]:
        return list(range(0, self.x_max + 1))

    def create_x_array(self) -> np.ndarray:
        nb_histograms = self.get_nb_histograms()
        return np.asarray(self.create_x_keys() * nb_histograms, dtype=int).reshape(
            (nb_histograms, self.x_max + 1)
        )


class Uniformise:
    @staticmethod
    def uniformise_histograms(histograms: List[Histogram]) -> HistogramsUniformised:
        """Uniformise distributions by finding the max key and adding zeros for all
        entries such that the distributions all have the same keys.
        """
        assert len(histograms) > 0, "list of histograms is empty"
        assert len(histograms) > 1, "found only one histogram"
        uniformised_hists = list()
        max_ = set([max(distr.keys()) for distr in histograms])
        the_max = max(list(max_))
        upper_bound = the_max + 1
        for hist in histograms:
            hist_ = list()
            for k in range(0, upper_bound):
                hist_.append(hist.get(k, 0))
            uniformised_hists.append(hist_)
        return HistogramsUniformised(
            the_max,
            np.array(uniformised_hists, dtype=int).reshape(
                (len(histograms), upper_bound)
            ),
        )

    @staticmethod
    def pooled_distribution(histograms: List[Histogram]) -> Distribution:
        """Create an averaged distribution by pulling all the histograms from
        different simulations together.
        >>> from src.futils import snapshot
        >>> histograms = [
        ...     snapshot.Histogram({0: 2, 1: 2}),
        ...     snapshot.Histogram({2: 1, 4: 1}),
        ... ]
        >>> distribution = snapshot.Uniformise.pooled_distribution(histograms)
        >>> list(distribution.keys())
        [0, 1, 2, 3, 4]
        >>> [round(ele, 2) for ele in distribution.values()]
        [0.33, 0.33, 0.17, 0.0, 0.17]
        """
        histograms_uniformed = Uniformise.uniformise_histograms(histograms)
        tot_cells = histograms_uniformed.y.sum()
        return Distribution(
            {
                k: val / tot_cells
                for k, val in zip(
                    range(0, histograms_uniformed.x_max + 1),
                    histograms_uniformed.y.sum(axis=0),
                )
            }
        )

    @staticmethod
    def pooled_histogram(histograms: List[Histogram]) -> Histogram:
        """Create an averaged histogram by pulling all the histograms from
        different simulations together without normalisation of the y axis.
        >>> from src.futils import snapshot
        >>> histograms = [
        ...     snapshot.Histogram({0: 2, 1: 2, 5: 1}),
        ...     snapshot.Histogram({2: 1, 4: 1, 5: 2}),
        ... ]
        >>> histrogram = snapshot.Uniformise.pooled_histogram(histograms)
        >>> list(histrogram.keys())
        [0, 1, 2, 3, 4, 5]
        >>> list(histrogram.values())
        [2, 2, 1, 0, 1, 3]
        """
        histograms_uniformed = Uniformise.uniformise_histograms(histograms)
        return Histogram(
            {
                k: val
                for k, val in zip(
                    range(0, histograms_uniformed.x_max + 1),
                    histograms_uniformed.y.sum(axis=0),
                )
            }
        )
