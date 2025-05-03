"""Module for plotting the results of the experiments."""

import pickle
from itertools import product
from pathlib import Path

import numpy as np
from sicore import SummaryFigure  # type: ignore[import]

from experiment.utils import Results


def plot_main(option: str, mode: str) -> None:
    """Plot the results of the experiments."""
    values: list[float]
    ylabel, is_null, num_seeds = "Type I Error Rate", True, 10
    fig_path = Path("figures/main") / f"fpr_{option}_{mode}.pdf"
    match mode:
        case "n":
            values = [100, 200, 300, 400]
            result_name = lambda value, seed: f"{value}_20_0.0_{seed}.pkl"
            xlabel = "number of samples"
        case "delta":
            values = [0.2, 0.4, 0.6, 0.8]
            result_name = lambda value, seed: f"200_20_{value}_{seed}.pkl"
            fig_path = Path("figures/main") / f"tpr_{option}.pdf"
            xlabel, ylabel, is_null, num_seeds = "true coefficient", "Power", False, 1

    figure = SummaryFigure(xlabel=xlabel, ylabel=ylabel)
    for value in values:
        results = Results()
        for seed in range(num_seeds):
            path = Path(f"results_{option}") / result_name(value, seed)
            with path.open("rb") as f:
                results += pickle.load(f)
        assert len(results) == num_seeds * 1000

        figure.add_results(results.results, label="proposed", xloc=value)
        figure.add_results(results.oc_p_values, label="w/o-pp", xloc=value)

        # Bonferroni correction
        n_: float
        d_: float
        match mode:
            case "n":
                n_, d_ = value, 20
            case "delta":
                n_, d_ = 200, 20

        figure.add_results(
            results.results,
            label="bonferroni",
            xloc=value,
            bonferroni=True,
            log_num_comparisons=(n_ + d_) * np.log(2),
        )

        if is_null:
            figure.add_results(results.results, label="naive", xloc=value, naive=True)

    if is_null:
        figure.add_red_line(value=0.05, label="significance level")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    figure.plot(fig_path, fontsize=20, legend_loc="upper left", yticks=[0.0, 0.5, 1.0])


if __name__ == "__main__":
    for option, mode in product(
        ["op1", "op2", "all_cv"],
        ["n", "delta"],
    ):
        plot_main(option, mode)
