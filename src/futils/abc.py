import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, NewType, Tuple


Posterior = NewType("Posterior", pd.Series)


# https://realpython.com/python-interface/
class Stat(ABC):
    @abstractmethod
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "distance")
            and callable(subclass.distance)
            or NotImplemented
        )


Stats = NewType("Stats", List[Stat])


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
    xlim = None
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
