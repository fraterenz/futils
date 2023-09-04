"""We load and process a histogram representing a single snapshot of a
simulation at a certain timepoint.
The histogram are stored as json files with keys being a quantity of interest
and the values are the number of individuals.
"""
import numpy as np
from typing import Dict, List, NewType


Distribution = NewType("Distribution", Dict[int, float])
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
        return np.asarray(
            list(range(0, self.x_max + 1)) * nb_histograms, dtype=int
        ).reshape((nb_histograms, self.x_max + 1))


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
