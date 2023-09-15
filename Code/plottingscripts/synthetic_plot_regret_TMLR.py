"""Generate the plots with times on synthetic data."""
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# from utils import get_function_evaluation_data as get_data
import parameters


from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=1, ncols=1))
plt.rcParams.update(figsizes.tmlr2023(nrows=1, ncols=2))
# plt.rcParams.update(figsizes.icml2022_half())

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


def get_ei_data(kernelname, filename, n):
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + "threedimse_jitter0.001"
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_ucb_data(kernelname, filename, n):
    regret_dframe = pd.read_csv(
        "./results/" + "groundtruth" + "/" + "['" + kernelname + "']" + "/" + filename
        # + "threedimse_beta100aistats"
        + "ucb_threedimse_discretization1000_tmlr" + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_adjusted_data(kernelname, filename):
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
            + "turbo_small_batch"
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
        + "betaexperiment_lengthscale_betaNonetmlr"
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_random_data(kernelname, filename):
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + "rebuttal_random"
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def get_adabkb_data(kernelname, filename):
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + "adabkb_threedimse_beta100_ucb_beta1_tmlr"
        + ".txt",
        sep="#",
        header=None,
        names=range(1000),
    )
    return regret_dframe


def get_direct_data(kernelname, filename, n):
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + "direct_regretdirect_threedimse_tmlr"
        + ".txt",
        sep="#",
        header=None,
        names=range(1000),
    )
    return regret_dframe


def plot_regret_helper(dframe, axis, steps, individual):
    """Helper function to plot the regret."""
    dataframe, label, color = dframe
    # print(dataframe, label)
    steps = 500  # np.minimum(steps, dataframe.shape[1])

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i][:steps]))

        min_simple_regret[min_simple_regret == 0] = 1 / (2 ** 10)

        min_simple_regret = np.log(min_simple_regret)

        average_min_regret += min_simple_regret
        min_simple_regret_list.append(min_simple_regret)
        alpha = 0.1
        if label == "AdaBkb":
            alpha = 0.1
        if individual:
            axis.plot(
                range(len(min_simple_regret)),
                np.asarray(min_simple_regret),
                linestyle="-",
                linewidth="1",
                alpha=alpha,
                color=color,
            )

    average_min_regret *= 1 / (dataframe.shape[0])

    if not individual:
        # plot the averade minimal regret
        if label == "AdaBkb":
            print("average min regret", np.asarray(average_min_regret))
        axis.plot(
            range(len(average_min_regret)),
            np.asarray(average_min_regret),
            linestyle="-",
            label=label,
            linewidth="2",
            color=color,
        )

    # axis.set_yscale("symlog", linthreshy=0.001)
    # axis.set_xscale("log")
    # axis.set_yscale("log")
    # axis.legend()


def plot_cumulative_regret_helper(dframe, axis, steps, individual):
    """Helper function to plot the regret."""
    dataframe, label, color = dframe
    # print(dataframe, label)
    steps = np.minimum(steps, dataframe.shape[1])
    steps = 500

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(
            np.add.accumulate(dataframe.iloc[i][:steps]) / np.arange(1, steps + 1)
        )
        average_min_regret += np.log(min_simple_regret)
        min_simple_regret_list.append(min_simple_regret)
        if individual:
            axis.plot(
                range(len(min_simple_regret)),
                np.asarray(np.log(min_simple_regret)),
                linestyle="-",
                linewidth="0.75",
                alpha=0.3,
                color=color,
            )

    average_min_regret *= 1 / (dataframe.shape[0])
    average_min_regret = np.asarray(average_min_regret)

    if not individual:
        # plot the averade minimal regret
        axis.plot(
            range(len(average_min_regret)),
            np.asarray(average_min_regret),
            linestyle="-",
            label=label,
            linewidth="2",
            color=color,
        )

    # axis.set_yscale("symlog", linthreshy=0.001)
    # axis.set_xscale("log")
    # axis.set_yscale("log")
    # axis.legend()


def plot_regret(kernelname, lengthscale, axis, steps, individual):
    """Plot the minmal regret for the specified kernel and lengthscale on the given axis."""
    names_and_colors = [
        # ("adjustedHOOregretadjusted_gpoo", "GP-OO (adjusted)", "red"),
        ("HOOregretlengthscalegpoo_", "GP-OO", GP_OO_COLOR),
        # ("ucb_regret", "GP-UCB", "blue"),
        # ("randomregret", "random", "green"),
        ("turboturbo_regret", "TurBO", TURBO_COLOR),
        # ("ei_regret", "TurBO", "yellow"),
    ]

    dframes = [
        (get_adjusted_data(kernelname, name), title, color)
        for name, title, color in names_and_colors
    ]

    dframes.append((get_ei_data(kernelname, "ei_regret", 1), "EI", EI_COLOR))
    dframes.append((get_ucb_data(kernelname, "_ucb_regret", 1), "GP-UCB", GP_UCB_COLOR))
    dframes.append(
        (get_direct_data(kernelname, "direct_regret", 1), "Direct", DIRECT_COLOR)
    )
    dframes.append(
        (get_random_data(kernelname, "randomregret"), "random", RANDOM_COLOR)
    )


    for dframe in dframes:
        plot_regret_helper(dframe, axis, steps, individual)


def plot_cumulative_regret(kernelname, lengthscale, axis, steps, individual):
    """Plot the minmal regret for the specified kernel and lengthscale on the given axis."""
    names_and_colors = [
        # ("adjustedHOOregretadjusted_gpoo", "GP-OO (adjusted)", "red"),
        ("HOOregretlengthscalegpoo_", "GP-OO", GP_OO_COLOR),
        # ("ucb_regret", "GP-UCB", "blue"),
        # ("randomregret", "random", "green"),
        ("turboturbo_regret", "TurBO", TURBO_COLOR),
        # ("ei_regret", "TurBO", "yellow"),
    ]

    dframes = [
        (get_adjusted_data(kernelname, name), title, color)
        for name, title, color in names_and_colors
    ]

    # dframes.append((get_turbo_data(), "Turbo", "blue"))
    dframes.append((get_ei_data(kernelname, "ei_regret", 1), "EI", EI_COLOR))
    dframes.append((get_ucb_data(kernelname, "_ucb_regret", 1), "GP-UCB", GP_UCB_COLOR))
    dframes.append(
        (get_direct_data(kernelname, "direct_regret", 1), "DiRect", DIRECT_COLOR)
    )


    for dframe in dframes:
        plot_cumulative_regret_helper(dframe, axis, steps, individual)


def make_plot():
    """Plot the minmal regret for all combinations of kernels and lengthscales."""
    kernelname1 = "squaredexponential"
    fig, axs = plt.subplots(1, 2)
    plot_regret(kernelname1, "0.2", axs[0], 100, individual=True)
    plot_cumulative_regret(kernelname1, "0.2", axs[1], 1000, individual=True)

    plot_regret(kernelname1, "0.2", axs[0], 100, individual=False)
    plot_cumulative_regret(kernelname1, "0.2", axs[1], 1000, individual=False)

    fig.suptitle("square exponential" + " 0.2")
    # add a legend on top
    handles, labels = axs[0].get_legend_handles_labels()
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
    axs[0].set_ylabel("$\log \,min_n\, r_n$")
    axs[1].set_ylabel("$\log \, R_n/n$")

    plt.savefig(
        "./plots/synthetic_regret_TMLR_rebuttal_with_DiRect.pdf",
        bbox_inches="tight",
    )
    plt.show()


make_plot()
