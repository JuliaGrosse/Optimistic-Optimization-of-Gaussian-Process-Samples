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

from plottingscripts.timing_results_rebuttal import plot_min_regret_per_time
import plottingscripts.timing_results_rebuttal as timing_results

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


def get_benchmark_filenames_ei(benchmark, i, jitter):
    return benchmark + "_domain" + str(i) + "_jitter" + str(jitter) + "_aistats"


def get_benchmark_filenames_ucb(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_aistats"


def get_benchmark_filenames_turbo(benchmark, i):
    return benchmark + "_domain" + str(i) + "_aistats_small_batch_verylong"


##def get_benchmark_filenames_adabkb(benchmark, i, beta):
#    return benchmark + "_domain" + str(i) + "_beta_" + str(beta) + "_tmlr"
def get_benchmark_filenames_direct(benchmark, i):
    # return benchmark + "_domain"+str(i)+"_aistats_rebuttal_long"
    return benchmark + "_domain" + str(i) + "direct" + "_tmlr_rebuttal_with_times2"


def get_benchmark_filenames_gpoo(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"


def get_benchmark_filenames_gpoo_more_steps(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"


def get_turbo_data(benchmark, experimentname, filename, filename_time, filename_evals):
    kernelname = "matern"
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
    return regret_dframe_filename, time_dframe_filename, evals_dframe_filename


def get_data(benchmark, experimentname, filename, filename_time):
    """Load the results for the given kernel and lengthscale."""
    kernelname = "matern"
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
    return regret_dframe_filename, time_dframe_filename


def plot_regret_helper(dframe, axis, steps, color, benchmark):
    """Helper function to plot the regret."""
    dataframe, label = dframe
    # print(dataframe, label)

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i])[:steps])
        average_min_regret += min_simple_regret
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
        average_min_regret,
        linestyle="-",
        linewidth=0.75,
        label=label,
        color=color,
    )
    axis.set_yscale("symlog", linthreshy=0.1)
    axis.set_xscale("log")


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


GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
DIRECT_COLOR = "blue"


# Final parameters obtained with a grid search. First entry: GP-UCB, Second entry: GP-OO


def read_turbo_times(filename, costs, batchsize=10, nb_init=25):
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            line = line[2:]
            line = line[:-2]
            words = line.split(",")
            for i, word in enumerate(words):
                iteration_time = float(word) + costs
                iteration_times.append(iteration_time)
            all_iteration_times.append(iteration_times)
    stored_times["iterations"] = all_iteration_times
    turbo_iterations = np.mean(np.asarray(stored_times["iterations"]), axis=0)
    times = np.cumsum(turbo_iterations)
    return times


def read_turbo_steps(evals_filename):
    evals_dframe = pd.read_csv(evals_filename, sep="#", header=None)
    return evals_dframe.iloc[0, :]


def plot_cumulative_time():
    """Plot the minimal regret per step for GP-OO and GP-UCB on the benchmark functions."""
    fig, axs = plt.subplots(4, 3)
    benchmarks_and_domains = list(benchmark_functions.ADAPTED_DOMAINS.items())[:12]
    for unpack, axs in zip(benchmarks_and_domains, axs.ravel()):
        benchmark, domain = unpack
        for i in range(NB_DOMAINS):

            id_gpucb = get_benchmark_filenames_ucb(
                benchmark, i, beta=GPUCB_BETA[benchmark]
            )
            id_ei = get_benchmark_filenames_ei(
                benchmark, i, jitter=EI_JITTER[benchmark]
            )
            id_turbo = get_benchmark_filenames_turbo(benchmark, i)

            id_direct = get_benchmark_filenames_direct(
                benchmark, i
            )

            regret_turbo, time_turbo, evals_turbo = get_turbo_data(
                benchmark,
                id_turbo,
                "turboturbo_regret",
                "turbotimelogsturbo",
                "turboevals",
            )
            axs.plot(
                np.asarray(read_turbo_steps(evals_turbo)),
                np.asarray(read_turbo_times(time_turbo, 0)),
                color=TURBO_COLOR,
                linewidth=1,
                label="TurBO",
            )

            regret_ucb, time_ucb = get_data(
                benchmark, id_gpucb, "ucb_regret", "loggingucb\n_"
            )
            regret_ei, time_ei = get_data(benchmark, id_ei, "ei_regret", "loggingei\n_")

            regret_direct, time_direct = get_data(
                benchmark, id_direct, "direct_regret", "directtimelogs"
            )
            time_data_direct = timing_results.get_time_data(time_direct)

            time_data_ei = timing_results.get_time_data(time_ei)
            time_data_gpucb = timing_results.get_time_data(time_ucb)

            axs.plot(time_data_ei, color=EI_COLOR, linewidth=1, label="EI")
            axs.plot(time_data_gpucb, color=GP_UCB_COLOR, linewidth=1, label="GP-UCB")
            axs.plot(time_data_direct, color="blue", linewidth=1, label="DiRect")

            id_gpoo = get_benchmark_filenames_gpoo(
                benchmark, i, beta=GPOO_BETA[benchmark]
            )
            regret_gpoo, time_gpoo = get_data(
                benchmark, id_gpoo, "HOOregretgreedy", "HOOtimelogsgreedy"
            )

            time_data_gpoo = timing_results.get_time_data(time_gpoo)
            axs.plot(
                time_data_gpoo,
                color=GP_OO_COLOR,
                linewidth=1,
                label="GP-OO",
            )

        axs.set_yscale("log")
        # axs.set_xscale("log")

        handles, labels = axs.get_legend_handles_labels()
        axs.set_title(
            DOMAIN_NAMES[benchmark] + " (dimension: " + str(len(domain)) + ")"
        )
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    fig.supxlabel("number of function evaluations")
    fig.supylabel("time in seconds")
    plt.savefig(
        "./plots/benchmarkplots/benchmark_cumulative_time_TMLR.pdf",
        bbox_inches="tight",
    )
    plt.show()


plot_cumulative_time()
