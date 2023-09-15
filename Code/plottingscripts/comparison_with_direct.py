"""Generate the plots with times on synthetic data."""
import random
import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt


# from utils import get_function_evaluation_data as get_data
import parameters


from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=3, ncols=3))
plt.rcParams.update(figsizes.tmlr2023(nrows=3, ncols=3, height_to_width_ratio=1.0))
# plt.rcParams.update(figsizes.icml2022_half())
# plt.rcParams.update(height_to_width_ratio=1.0)

random.seed(28)
np.random.seed(28)


GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
RANDOM_COLOR = "black"
DIRECT_COLOR = "blue"


def get_turbo_data(turbo_name):
    regret_dframe = pd.read_csv(turbo_name, sep="#", header=None)

    def cut(x):
        return x[1:-1]

    regret_dframe = regret_dframe.applymap(cut)
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_random_data(kernelname, filename, dimension, lengthscale, steps):
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + "comparison_dim"
        + str(dimension)
        + "_lengthscale_"
        + str(lengthscale)
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_adjusted_data(kernelname, filename, dimension, lengthscale):
    if "turbo" in filename:
        filename = (
            "./results/"
            + "groundtruth"
            + "/"
            + "['"
            + kernelname
            + "']"
            + "/"
            + filename
            # + "threedimseturbo"
            + "turbo_comparison_dim"
            + str(dimension)
            + "_lengthscale_"
            + str(lengthscale)
            + ".txt"
        )
        return get_turbo_data(filename)
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + "more_accurate_comparison_dim"
        + str(dimension)
        + "_lengthscale_"
        + str(lengthscale)
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_direct_data(kernelname, filename, dimension, lengthscale, steps):
    regret_dframe = pd.read_csv(
        "./results/" + "groundtruth" + "/" + "['" + kernelname + "']" + "/"
        # + "direct_regretdirect_comparison_dim"
        + "direct_regretcomparison_dim"
        + str(dimension)
        + "_lengthscale_"
        + str(lengthscale)
        + ".txt",
        sep="#",
        header=None,
        names=range(steps),
    )
    return regret_dframe


def get_mins_and_max(dimension, lengthscale):
    samples = np.load(
        # "/Users/juliagrosse/Desktop/FastBO/TMLR/Code/results/groundtruth/['matern']/samples_direct_comparison_dim"
        "/Users/juliagrosse/Desktop/FastBO/TMLR/Code/results/groundtruth/['squaredexponential']/samples_comparison_dim"
        + str(dimension)
        + "_lengthscale_"
        + str(lengthscale)
        + ".npy"
    )
    minimas, maximas = [], []
    for sample in samples:
        minimas.append(np.min(sample))
        maximas.append(np.max(sample))
    return minimas, maximas


def plot_regret_helper(dframe, axis, steps, individual, dimension, lengthscale):
    """Helper function to plot the regret."""
    dataframe, label, color = dframe
    steps = np.minimum(steps, dataframe.shape[1])

    minimas, maximas = get_mins_and_max(dimension, lengthscale)
    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        # print("regret i",dataframe.iloc[i][:steps]  )
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i][:steps]))
        print("Min simple regret", np.asarray(min_simple_regret)[:20])
        # min_simple_regret = minimas[i] + mi
        # min_simple_regret -= minimas[i]
        min_simple_regret *= 1 / (maximas[i] - minimas[i])
        # min_simple_regret += 1e-10

        # min_simple_regret[min_simple_regret <=1/(2**100)] = 1 / (2 ** 100)

        min_simple_regret = min_simple_regret

        average_min_regret += min_simple_regret
        min_simple_regret_list.append(min_simple_regret)
        alpha = 0.2
        if individual:
            axis.plot(
                range(1, len(min_simple_regret) + 1),
                np.asarray(min_simple_regret),
                linestyle="-",
                linewidth="2",
                alpha=alpha,
                color=color,
            )

    average_min_regret *= 1 / (dataframe.shape[0])

    if not individual:
        # plot the averade minimal regret
        axis.plot(
            range(1, len(average_min_regret) + 1),
            np.asarray(average_min_regret),
            linestyle="-",
            label=label,
            linewidth="2",
            color=color,
        )
    axis.set_title("dim: " + str(dimension) + ", l: " + str(lengthscale))
    axis.set_yscale("symlog", linthreshy=0.1)
    # axis.set_xscale("log")
    # axis.set_yscale("logit")
    # axis.legend()


def plot_regret(
    kernelname, dimension, lengthscale, axis, steps, plot_steps, individual
):
    """Plot the minmal regret for the specified kernel and lengthscale on the given axis."""
    names_and_colors = [
        ("HOOregretlengthscalegpoo_", "GP-OO", GP_OO_COLOR),
        # ("turbo_regret", "TurBO", TURBO_COLOR)
    ]

    dframes = []
    dframes.append(
        (
            get_direct_data(kernelname, "direct_regret", dimension, lengthscale, steps),
            "Direct",
            DIRECT_COLOR,
        )
    )
    # dframes.append(
    #     (
    #         get_random_data(kernelname, "randomregret", dimension, lengthscale, steps),
    #         "random",
    #         RANDOM_COLOR,
    #     )
    # )

    dframes += [
        (get_adjusted_data(kernelname, name, dimension, lengthscale), title, color)
        for name, title, color in names_and_colors
    ]

    for dframe in dframes:
        plot_regret_helper(dframe, axis, plot_steps, individual, dimension, lengthscale)


def make_plot():
    """Plot the minmal regret for all combinations of kernels and lengthscales."""
    kernelname1 = "squaredexponential"
    fig, axs = plt.subplots(3, 3)

    ## Experiments in 3 dim
    plot_regret(kernelname1, 3, 1, axs[0, 0], 1000, 1000, individual=True)
    plot_regret(kernelname1, 3, 1, axs[0, 0], 1000, 1000, individual=False)

    plot_regret(kernelname1, 3, 0.1, axs[0, 1], 10000, 5000, individual=True)
    plot_regret(kernelname1, 3, 0.1, axs[0, 1], 10000, 5000, individual=False)

    plot_regret(kernelname1, 3, 0.05, axs[0, 2], 100000, 5000, individual=True)
    plot_regret(kernelname1, 3, 0.05, axs[0, 2], 100000, 5000, individual=False)
    #
    # ## Experiments in 2 dim
    plot_regret(kernelname1, 2, 0.5, axs[1, 0], 1000, 200, individual=True)
    plot_regret(kernelname1, 2, 0.5, axs[1, 0], 1000, 200, individual=False)

    plot_regret(kernelname1, 2, 0.05, axs[1, 1], 10000, 1000, individual=True)
    plot_regret(kernelname1, 2, 0.05, axs[1, 1], 10000, 1000, individual=False)

    plot_regret(kernelname1, 2, 0.005, axs[1, 2], 100000, 100000, individual=True)
    plot_regret(kernelname1, 2, 0.005, axs[1, 2], 100000, 100000, individual=False)
    #
    # ## Experiments in 1 dim
    plot_regret(kernelname1, 1, 0.5, axs[2, 0], 1000, 50, individual=True)
    plot_regret(kernelname1, 1, 0.5, axs[2, 0], 1000, 50, individual=False)
    #
    plot_regret(kernelname1, 1, 0.05, axs[2, 1], 10000, 200, individual=True)
    plot_regret(kernelname1, 1, 0.05, axs[2, 1], 10000, 200, individual=False)
    #
    plot_regret(kernelname1, 1, 0.005, axs[2, 2], 100000, 1000, individual=True)
    plot_regret(kernelname1, 1, 0.005, axs[2, 2], 100000, 1000, individual=False)

    # fig.suptitle("square exponential" + " 0.2")
    # add a legend on top
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="lower left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=4,
    # )

    # add axis labels
    fig.legend(handles, labels, loc="upper right")
    fig.supxlabel("number of function evaluations n")
    fig.supylabel("$(\log) \,min_n\, r_n$")
    if kernelname1 == "matern":
        fig.suptitle("Matern")
    else:
        fig.suptitle("squared exponential")

    # save and sho
    # +´
    # plt2tikz.save("./plots/regret_adjusted_centers.tex")
    plt.savefig(
        "./plots/comparison_with_Direct_regret_TMLR_rebuttal_more_accurate"
        + kernelname1
        + ".pdf",
        bbox_inches="tight",
    )
    plt.show()


make_plot()
