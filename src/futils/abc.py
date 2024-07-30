import pandas as pd
import numpy as np
from . import snapshot
from abc import ABC, abstractmethod
from scipy import stats
from typing import List, NewType, Set, Tuple


PosteriorIdx = NewType("PosteriorIdx", Set[int])
Posterior = NewType("Posterior", pd.Series)


# https://realpython.com/python-interface/
class Stat(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "distance")
            and callable(subclass.distance)
            or NotImplemented
        )

    @abstractmethod
    def distance(
        self, target: snapshot.Histogram, simulation: snapshot.Histogram
    ) -> float:
        raise NotImplementedError


Stats = NewType("Stats", List[Stat])


def filter_runs_stat(
    summary: pd.DataFrame, quantile: float, stat: Stat
) -> PosteriorIdx:
    stat_name = stat.__class__.__name__
    assert stat_name in set(
        summary.columns
    ), f"metric {stat_name} not found in df with cols {set(summary.columns)}"
    idx = summary.loc[
        summary[stat_name] <= summary[stat_name].quantile(quantile), "idx"
    ]
    idx_set = set(idx.idx.unique())
    assert idx.shape[0] == len(idx_set)
    return PosteriorIdx(idx_set)


@Stat.register
class Wasserstein:
    def __init__(self) -> None:
        super().__init__()

    def distance(self, target: snapshot.Histogram, sim: snapshot.Histogram) -> float:
        # uniformise such that they have the same support which is required by
        # the wasserstein metric
        target_uniformised, sim_uniformised = snapshot.Uniformise.uniformise_histograms(
            [target, sim]
        ).make_histograms()

        assert len(target_uniformised) == len(sim_uniformised)

        v_values, v_weights = list(sim_uniformised.keys()), list(
            sim_uniformised.values()
        )
        u_values, u_weights = list(target_uniformised.keys()), list(
            target_uniformised.values()
        )
        return stats.wasserstein_distance(u_values, v_values, u_weights, v_weights)


def round_estimates(estimate: float, significant: str) -> str:
    if significant == "three":
        return str(round(estimate, 3))
    elif significant == "two":
        return str(round(estimate, 2))
    elif significant == "one":
        return str(round(estimate, 1))
    elif significant == "zero":
        return str(int(round(estimate, 0)))
    raise ValueError(f"significant must be 'two' 'one' or 'zero', not '{significant}'")


class Estimate:
    def __init__(
        self,
        name: str,
        point_estimate,
        credible_interval_90: Tuple[float, float],
    ):
        """MAP estimate with 90% credibility interval"""
        self.name = name
        self.point_estimate = point_estimate
        if point_estimate < credible_interval_90[0]:
            self.credible_interval_90 = (
                point_estimate,
                credible_interval_90[1],
            )
        else:
            self.credible_interval_90 = credible_interval_90

    def to_string(self, precision: str) -> str:
        point_estimate = round_estimates(self.point_estimate, precision)
        interval = round_estimates(
            self.point_estimate - self.credible_interval_90[0], precision
        ), round_estimates(
            self.credible_interval_90[1] - self.point_estimate, precision
        )
        return f"{point_estimate}^{{+{interval[1]}}}_{{-{interval[0]}}}"


class Bin:
    def __init__(self, name: str, bin_: np.ndarray):
        self.name = name
        self.bin = bin_
        self.bin_distance = (self.bin[1] - self.bin[0]) / 2

    def compute_hist(self, posterior: Posterior) -> np.ndarray:
        values, _ = np.histogram(posterior, bins=self.bin, density=True)
        return values

    def compute_estimate(self, posterior: Posterior) -> Estimate:
        values = self.compute_hist(posterior)
        point_estimate = self.bin[np.argmax(values)] + self.bin_distance
        return Estimate(
            self.name,
            point_estimate,
            (posterior.quantile((0.10)), posterior.quantile(0.90)),
        )


def plot_posterior(
    posterior: Posterior,
    xlabel: str,
    bins: Bin,
    ax,
    color,
    fancy: bool,
    legend: bool = False,
    xlim=None,
):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/stairs_demo.html
    values = bins.compute_hist(posterior)
    if fancy:
        ax.fill_between(
            bins.bin[:-1] + np.diff(bins.bin) / 2,
            values,
            ls="-",
            color=color,
            alpha=0.3,
        )
        ax.plot(
            bins.bin[:-1] + np.diff(bins.bin) / 2,
            values,
            ls="-",
            marker=".",
            mew=3,
            color=color,
            alpha=0.5,
        )
    else:
        values, bins, _ = ax.hist(
            posterior,
            align="mid",
            alpha=0.4,
            density=True,
            bins=bins.bin,
            edgecolor="black",
            color=color,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pdf")
    if xlim:
        ax.set_xlim(xlim)
    if legend:
        ax.legend()
    return ax
