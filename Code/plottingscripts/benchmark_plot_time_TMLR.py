"""Plot the results from the benchmark experiments."""
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json


from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=4, ncols=3))
plt.rcParams.update(figsizes.tmlr2023(nrows=4, ncols=3, height_to_width_ratio=1))

import benchmark_functions

from plottingscripts.timing_results_rebuttal_logscale import plot_min_regret_per_time

random.seed(28)

NB_DOMAINS = 10


DOMAIN_NAMES = {
    "sixhumpcamel": "Six-Hump Camel",
    "dixonprice10": "Dixon-Price",
    "branin": "Branin",
    "beale": "Beale",
    "bohachevsky_a": "Bohachevsky A",
    "bohachevsky_b": "Bohachevsky B",
    "bohachevsky_c": "Bohachevsky C",
    "rosenbrock2": "Rosenbrock",
    "ackley2": "Ackley",
    "hartmann3": "Hartmann",
    "trid4": "Trid",
    "shekel": "Shekel",
}

EI_JITTER = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/ei_jitter.json")
)
GPUCB_BETA = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/gpucb_beta.json")
)
GPOO_BETA = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/gpoo_beta.json")
)

ADABKB_BETA = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/adabkb_beta.json")
)


def get_benchmark_filenames_adabkb(benchmark, beta):
    return [
        benchmark + "_domain" + str(i) + "_beta_" + str(beta) + "_tmlr"
        for i in range(10)
    ]

def get_benchmark_filenames_direct(benchmark):
    return [benchmark + "_domain" + str(i) + "direct" + "_tmlr_rebuttal_with_times2" for i in range(10)]


def get_benchmark_filenames_ei(benchmark, jitter):
    return [
        benchmark + "_domain" + str(i) + "_jitter" + str(jitter) + "_aistats"
        for i in range(10)
    ]


def get_benchmark_filenames_ucb(benchmark, beta):
    return [
        benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_aistats"
        for i in range(10)
    ]


def get_benchmark_filenames_random(
    benchmark,
):
    return [
        benchmark + "_domain" + str(i) + "_random_rebuttal_very_long" for i in range(10)
    ]


def get_benchmark_filenames_turbo(benchmark):
    return [
        benchmark + "_domain" + str(i) + "_aistats_small_batch_verylong"
        for i in range(10)
    ]


def get_benchmark_filenames_gpoo(benchmark, beta):
    if benchmark in ["dixonprice10", "shekel", "trid4"]:
        return [
            benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"
            for i in range(10)
        ]
    return [
        benchmark
        + "_domain"
        + str(i)
        + "_beta"
        + str(beta)
        + "_tmlr_rebuttal_good_implementation"
        for i in range(10)
    ]


def get_turbo_data(benchmark, experimentnames, filename, filename_time, filename_evals):
    kernelname = "matern"
    regret_dframe_filenames, time_dframe_filenames, evals_dframe_filenames = [], [], []

    for experimentname in experimentnames:
        regret_dframe_filename = (
            "./results/"
            + benchmark
            + "/['"
            + kernelname
            + "']/"
            + filename
            + experimentname
            + ".txt"
        )
        time_dframe_filename = (
            "./results/"
            + benchmark
            + "/['"
            + kernelname
            + "']/"
            + filename_time
            + experimentname
            + ".txt"
        )
        evals_dframe_filename = (
            "./results/"
            + benchmark
            + "/['"
            + kernelname
            + "']/"
            + filename_evals
            + experimentname
            + ".txt"
        )

        regret_dframe_filenames.append(regret_dframe_filename)
        time_dframe_filenames.append(time_dframe_filename)
        evals_dframe_filenames.append(evals_dframe_filename)
    return regret_dframe_filenames, time_dframe_filenames, evals_dframe_filenames


def get_data(benchmark, experimentnames, filename, filename_time):
    """Load the results for the given kernel and lengthscale."""
    kernelname = "matern"
    regret_dframe_filenames, time_dframe_filenames = [], []
    for experimentname in experimentnames:
        regret_dframe_filename = "/" + filename + experimentname + ".txt"
        time_dframe_filename = (
            "./results/"
            + benchmark
            + "/['"
            + kernelname
            + "']/"
            + filename_time
            + experimentname
            + ".txt"
        )
        regret_dframe_filenames.append(regret_dframe_filename)
        time_dframe_filenames.append(time_dframe_filename)
    return regret_dframe_filenames, time_dframe_filenames


def plot_regret_helper(dframe, axis, steps, color, benchmark):
    """Helper function to plot the regret."""
    dataframe, label = dframe
    # print(dataframe, label)

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i])[:steps])
        average_min_regret += np.log(min_simple_regret)
        min_simple_regret_list.append(min_simple_regret)

    average_min_regret *= 1 / (dataframe.shape[0])

    true_min = benchmark_functions.MINIMA[benchmark][1]
    axis.axhline(y=true_min, color="black", linewidth=0.5)

    # create list with standard deviation
    stds = []
    for i in range(steps):
        std = np.std([results[i] for results in min_simple_regret_list])
        stds.append(std)

    # plot the averade minimal regret
    axis.plot(
        range(len(average_min_regret)),
        -(true_min - average_min_regret),
        linestyle="-",
        linewidth=0.75,
        label=label,
        color=color,
    )
    # axis.set_yscale("symlog", linthreshy=0.1)
    # axis.set_yscale("log")
    axis.set_xscale("log")


GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
RANDOM_COLOR = "black"
DIRECT_COLOR = "blue"


# Final parameters obtained with a grid search. First entry: GP-UCB, Second entry: GP-OO


def plot_time_per_steps(costs=0):
    """Plot the minimal regret per step for GP-OO and GP-UCB on the benchmark functions."""
    fig, axs = plt.subplots(4, 3)
    benchmarks_and_domains = list(benchmark_functions.ADAPTED_DOMAINS.items())[:12]
    for unpack, axs in zip(benchmarks_and_domains, axs.ravel()):
        benchmark, domain = unpack
        true_min = benchmark_functions.MINIMA[benchmark][1]

        id_gpucb = get_benchmark_filenames_ucb(benchmark, beta=GPUCB_BETA[benchmark])
        id_ei = get_benchmark_filenames_ei(benchmark, jitter=EI_JITTER[benchmark])
        id_turbo = get_benchmark_filenames_turbo(benchmark)
        id_random = get_benchmark_filenames_random(benchmark)
        id_direct = get_benchmark_filenames_direct(benchmark)

        regret_ucb, time_ucb = get_data(
            benchmark, id_gpucb, "ucb_regret", "loggingucb\n_"
        )
        regret_ei, time_ei = get_data(benchmark, id_ei, "ei_regret", "loggingei\n_")
        regret_turbo, time_turbo, evals_turbo = get_turbo_data(
            benchmark, id_turbo, "turboturbo_regret", "turbotimelogsturbo", "turboevals"
        )
        regret_random, time_random = get_data(
            benchmark, id_random, "randomregret", "randomtimelogs"
        )

        regret_direct, time_direct = get_data(
            benchmark, id_direct, "direct_regret", "directtimelogs"
        )

        id_gpoo = get_benchmark_filenames_gpoo(benchmark, beta=GPOO_BETA[benchmark])
        regret_gpoo, time_gpoo = get_data(
            benchmark, id_gpoo, "HOOregretgreedy", "HOOtimelogsgreedy"
        )

        plot_min_regret_per_time(
            [regret_ei, time_ei],
            axs,
            200,
            "ei",
            "matern",
            None,
            benchmark,
            EI_COLOR,
            costs=costs,
        )
        plot_min_regret_per_time(
            [regret_ucb, time_ucb],
            axs,
            200,
            "ucb",
            "matern",
            None,
            benchmark,
            GP_UCB_COLOR,
            costs=costs,
        )
        plot_min_regret_per_time(
            (regret_turbo, time_turbo, evals_turbo),
            axs,
            10000,
            "turbo",
            "matern",
            None,
            benchmark,
            TURBO_COLOR,
            costs=costs,
        )
        plot_min_regret_per_time(
            (regret_direct, time_direct),
            axs,
            10000,
            "direct",
            "matern",
            None,
            benchmark,
            DIRECT_COLOR,
            costs=costs,
        )
        plot_min_regret_per_time(
            (regret_random, time_random),
            axs,
            10000,
            "random",
            "matern",
            None,
            benchmark,
            RANDOM_COLOR,
            costs=costs,
            hoo_batch_size=2,
        )

        if len(domain) < 4:
            plot_min_regret_per_time(
                (regret_gpoo, time_gpoo),
                axs,
                10000,
                "gpoo",
                "matern",
                None,
                benchmark,
                GP_OO_COLOR,
                costs=costs,
                hoo_batch_size=2,
            )
        elif len(domain) < 6:
            plot_min_regret_per_time(
                (regret_gpoo, time_gpoo),
                axs,
                10000,
                "gpoo",
                "matern",
                None,
                benchmark,
                GP_OO_COLOR,
                costs=costs,
                hoo_batch_size=2,
            )
        else:
            plot_min_regret_per_time(
                (regret_gpoo, time_gpoo),
                axs,
                10000,
                "gpoo",
                "matern",
                None,
                benchmark,
                GP_OO_COLOR,
                costs=costs,
                hoo_batch_size=2,
            )

        handles, labels = axs.get_legend_handles_labels()
        axs.set_title(
            DOMAIN_NAMES[benchmark] + " (dimension: " + str(len(domain)) + ")"
        )
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.supxlabel("log time in seconds")
    fig.supylabel("log of minimal simple regret")
    plt.savefig(
        "./plots/benchmarkplots/timecosts/benchmark_time_TMLR_costs_rebuttal"
        + str(costs)
        + ".pdf",
        bbox_inches="tight",
    )
    # plt.show()


plot_time_per_steps(costs=1)
plot_time_per_steps(costs=0.1)
plot_time_per_steps(costs=0.01)
plot_time_per_steps(costs=0.001)
