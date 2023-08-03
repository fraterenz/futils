"""We load and process a histogram representing a single snapshot of a
simulation at a certain timepoint.
The histogram are stored as json files with keys being a quantity of interest
and the values are the number of individuals.
"""
import numpy as np
from typing import Dict, List, NewType


Distribution = NewType("Distribution", Dict[int, int])
Histogram = NewType("Histogram", Dict[int, int])


class HistogramsUniformised:
    x_max: int
    y: np.ndarray  # shape experiments x x_max

    def __init__(self, x_max: int, y: np.ndarray) -> None:
        self.x_max = x_max
        self.y = y

    def make_distribution(self) -> Distribution:
        raise NotImplementedError

    def create_x_array(self) -> np.ndarray:
        nb_histograms = self.y.shape[0]
        return np.asarray(list(range(0, self.x_max + 1)) * nb_histograms, dtype=int).reshape((nb_histograms, self.x_max + 1) )


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
