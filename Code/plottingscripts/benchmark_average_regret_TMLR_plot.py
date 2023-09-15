"""Plot the results from the benchmark experiments."""
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import json

import benchmark_functions

random.seed(28)

NB_DOMAINS = 10

from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=4, ncols=3))
plt.rcParams.update(figsizes.tmlr2023(nrows=4, ncols=3, height_to_width_ratio=1))


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

BENCHMARK_MINIMA = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/minima.txt")
)


def get_benchmark_filenames_ei(benchmark, i, jitter):
    return benchmark + "_domain" + str(i) + "_jitter" + str(jitter) + "_aistats"


def get_benchmark_filenames_random(benchmark, i):
    return benchmark + "_domain" + str(i) + "_random_rebuttal_very_long"


def get_benchmark_filenames_ucb(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_aistats"


def get_benchmark_filenames_turbo(benchmark, i):
    # return benchmark + "_domain"+str(i)+"_aistats_rebuttal_long"
    return benchmark + "_domain" + str(i) + "_aistats_small_batch_verylong"


def get_benchmark_filenames_adabkb(benchmark, i, beta):
    # return benchmark + "_domain"+str(i)+"_aistats_rebuttal_long"
    return (
        benchmark
        + "_domain"
        + str(i)
        + "_gpoo_beta_"
        + str(beta)
        + "_gpucb_beta1"
        + "_tmlr_rebuttal"
    )


def get_benchmark_filenames_direct(benchmark, i):
    # return benchmark + "_domain"+str(i)+"_aistats_rebuttal_long"
    return benchmark + "_domain" + str(i) + "direct" + "_tmlr_rebuttal"


def get_benchmark_filenames_gpoo(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"


def get_turbo_data(benchmark, experimentname, filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        sep="#",
        header=None,
    )

    def cut(x):
        return x[1:-1]

    regret_dframe = regret_dframe.applymap(cut)
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_ada_bkb_data(benchmark, experimentname, filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        header=None,
        sep="#",
        names=range(10000),
    )
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_direct_data(benchmark, experimentname, filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        header=None,
        sep="#",
        names=range(10000),
    )
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_data(benchmark, experimentname, filename):
    """Load the results for the given kernel and lengthscale."""
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        sep="#",
        header=None,
    )
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def plot_average_regret_helper_adabkb(dframe, axis, steps, color, benchmark):
    """Helper function to plot the regret."""
    dataframe, label = dframe
    # print(dataframe, label)

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    true_min = benchmark_functions.MINIMA[benchmark][1]
    for i in range(dataframe.shape[0]):
        min_function_values = np.squeeze(
            np.minimum.accumulate(dataframe.iloc[i])[:steps]
        )
        # min_simple_regret = np.squeeze(np.cumsum(dataframe.iloc[i])[:steps])
        min_simple_regret = -(true_min - min_function_values)
        min_simple_regret[min_simple_regret <= 0] = 1 / 2 ** 20
        scaled_min_simple_regret = np.log(min_simple_regret)
        axis.plot(
            np.asarray(range(len(min_simple_regret))),
            np.asarray(scaled_min_simple_regret),
            linestyle="-",
            linewidth=1,
            alpha=1,
            color=color,
        )
    axis.set_xscale("log")


def plot_average_regret_helper(dframe, axis, steps, color, benchmark):
    """Helper function to plot the regret."""
    dataframe, label = dframe
    # print(dataframe, label)

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    true_min = benchmark_functions.MINIMA[benchmark][1]
    for i in range(dataframe.shape[0]):
        min_function_values = np.squeeze(
            np.minimum.accumulate(dataframe.iloc[i])[:steps]
        )
        # min_simple_regret = np.squeeze(np.cumsum(dataframe.iloc[i])[:steps])
        min_simple_regret = -(true_min - min_function_values)
        min_simple_regret[min_simple_regret <= 1 / 2 ** 20] = 1 / 2 ** 20
        scaled_min_simple_regret = np.log(min_simple_regret)
        scaled_positive_min_simple_regret = np.log(min_simple_regret)
        average_min_regret += scaled_positive_min_simple_regret
        print("average_min_regret", label, scaled_min_simple_regret.shape)
        axis.plot(
            np.asarray(range(len(min_simple_regret))),
            np.asarray(scaled_min_simple_regret),
            linestyle="-",
            linewidth=1,
            alpha=0.2,
            color=color,
        )

    average_min_regret *= 1 / (dataframe.shape[0])

    # plot the averade minimal regret
    axis.plot(
        np.asarray(range(len(average_min_regret))),
        np.asarray(average_min_regret),
        linestyle="-",
        linewidth=2,
        alpha=1,
        label=label,
        color=color,
    )
    # axis.set_yscale("symlog", linthresh=0.0001)
    # axis.set_yscale("log")
    axis.set_xscale("log")


def average_over_all_domains(benchmark, domain):

    (
        regrets_ucb,
        regrets_ei,
        regrets_turbo,
        regrets_random,
        regrets_gpoo,
        regrets_adabkb,
        regrets_direct,
    ) = ([], [], [], [], [], [], [])

    for i in range(10):
        id_gpucb = get_benchmark_filenames_ucb(benchmark, i, beta=GPUCB_BETA[benchmark])
        id_ei = get_benchmark_filenames_ei(benchmark, i, jitter=EI_JITTER[benchmark])
        id_turbo = get_benchmark_filenames_turbo(benchmark, i)
        id_random = get_benchmark_filenames_random(benchmark, i)
        id_adabkb = get_benchmark_filenames_adabkb(
            benchmark, i, beta=ADABKB_BETA[benchmark]
        )
        id_direct = get_benchmark_filenames_direct(benchmark, i)

        id_gpoo = get_benchmark_filenames_gpoo(benchmark, i, beta=GPOO_BETA[benchmark])

        regret_ucb = get_data(benchmark, id_gpucb, "ucb_regret")
        regret_ei = get_data(benchmark, id_ei, "ei_regret")
        regret_turbo = get_turbo_data(benchmark, id_turbo, "turboturbo_regret")
        regret_random = get_data(benchmark, id_random, "randomregret")
        regret_gpoo = get_data(benchmark, id_gpoo, "HOOregretgreedy")
        regret_adabkb = get_ada_bkb_data(benchmark, id_adabkb, "adabkb_regret")
        regret_direct = get_direct_data(benchmark, id_direct, "direct_regret")

        regrets_ucb.append(regret_ucb)
        regrets_ei.append(regret_ei)
        regrets_turbo.append(regret_turbo)
        regrets_random.append(regret_random)
        regrets_gpoo.append(regret_gpoo)
        regrets_adabkb.append(regret_adabkb)
        regrets_direct.append(regret_direct)

    return (
        pd.concat(regrets_ucb),
        pd.concat(regrets_ei),
        pd.concat(regrets_turbo),
        pd.concat(regrets_random),
        pd.concat(regrets_gpoo),
        pd.concat(regrets_adabkb),
        pd.concat(regrets_direct),
    )


GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
RANDOM_COLOR = "black"
ADABKB_COLOR = "pink"
DIRECT_COLOR = "blue"


def plot_average_regret_per_steps():
    """Plot the minimal regret per step for GP-OO and GP-UCB on the benchmark functions."""
    fig, axs = plt.subplots(4, 3)
    benchmarks_and_domains = list(benchmark_functions.ADAPTED_DOMAINS.items())[:12]
    for unpack, axs in zip(benchmarks_and_domains, axs.ravel()):
        benchmark, domain = unpack

        (
            regret_ucb,
            regret_ei,
            regret_turbo,
            regret_random,
            regret_gpoo,
            regret_adabkb,
            regret_direct,
        ) = average_over_all_domains(benchmark, domain)
        print("benchmark", benchmark)

        plot_average_regret_helper(
            (regret_ei, "EI"), axs, 200, color=EI_COLOR, benchmark=benchmark
        )
        plot_average_regret_helper(
            (regret_ucb, "GP-UCB"), axs, 200, color=GP_UCB_COLOR, benchmark=benchmark
        )
        plot_average_regret_helper(
            (regret_turbo, "TurBO"), axs, 10000, color=TURBO_COLOR, benchmark=benchmark
        )
        plot_average_regret_helper(
            (regret_direct, "Direct"),
            axs,
            10000,
            color=DIRECT_COLOR,
            benchmark=benchmark,
        )
        # plot_average_regret_helper_adabkb(
        #     (regret_adabkb, "AdaBkb"),
        #     axs,
        #     10000,
        #     color=ADABKB_COLOR,
        #     benchmark=benchmark,
        # )

        plot_average_regret_helper(
            (regret_random, "random"),
            axs,
            10000,
            color=RANDOM_COLOR,
            benchmark=benchmark,
        )

        plot_average_regret_helper(
            (regret_gpoo, "GP-OO"),
            axs,
            regret_gpoo.shape[-1],
            color=GP_OO_COLOR,
            benchmark=benchmark,
        )

        handles, labels = axs.get_legend_handles_labels()
        # axs.set_title(benchmark)
        axs.set_title(
            DOMAIN_NAMES[benchmark] + " (dimension: " + str(len(domain)) + ")"
        )
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.supxlabel("number of function evaluations")
    fig.supylabel("log of minimal simple regret")
    plt.savefig(
        "./plots/benchmarkplots/benchmark_average_regret_TMLR_with_direct.pdf",
        bbox_inches="tight",
    )
    plt.show()


plot_average_regret_per_steps()
