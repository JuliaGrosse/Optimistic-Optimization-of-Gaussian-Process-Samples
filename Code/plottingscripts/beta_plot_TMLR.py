"""Generate the plots with times on synthetic data."""
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from tueplots import cycler
from tueplots.constants.color import palettes
from tueplots import figsizes, fonts, fontsizes

# from utils import get_function_evaluation_data as get_data
import parameters

from tueplots import axes, bundles

plt.rcParams.update(bundles.tmlr2023(nrows=1, ncols=2))

random.seed(28)
np.random.seed(28)

GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
RANDOM_COLOR = "black"

GREEDY_COLOR1 = "#F9521E"
GREEDY_COLOR2 = "#F9525E"
GREEDY_COLOR3 = "#D133FF"
GREEDY_COLOR4 = "#FF3377"


def get_data(kernelname, mode, beta):
    regret_dframe = pd.read_csv(
        "./results/" + "groundtruth" + "/" + "['" + kernelname + "']" + "/"
        "HOOregret"
        + mode
        + "gpoo_betaexperiment_"
        + mode
        + "_beta"
        + str(beta)
        + "tmlr"
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def plot_regret_helper(dframe, axis, steps, individual):
    """Helper function to plot the regret."""
    dataframe, label, color = dframe
    print(dataframe, label)
    steps = np.minimum(steps, dataframe.shape[1])

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i][:steps]))

        min_simple_regret[min_simple_regret == 0] = 1 / (2 ** 10)

        min_simple_regret = np.log(min_simple_regret)

        average_min_regret += min_simple_regret
        min_simple_regret_list.append(min_simple_regret)
        if individual:
            axis.plot(
                range(len(min_simple_regret)),
                min_simple_regret,
                linestyle="-",
                linewidth="0.75",
                alpha=0.3,
                color=color,
            )

    average_min_regret *= 1 / (dataframe.shape[0])

    if not individual:
        # plot the averade minimal regret
        axis.plot(
            range(len(average_min_regret)),
            average_min_regret,
            linestyle="-",
            label=label,
            linewidth="2",
            color=color,
        )


def plot_regret(kernelname, lengthscale, axis, steps, individual):
    """Plot the minmal regret for the specified kernel and lengthscale on the given axis."""
    names_and_colors = [
        ("lengthscale", None, "heuristic 1", GP_OO_COLOR),
        ("heuristic", None, "heuristic 2", EI_COLOR),
        ("discretization", None, "heuristic 3", TURBO_COLOR),
        ("greedy", 0.1, "0.1", GREEDY_COLOR1),
        ("greedy", 1, "1", GREEDY_COLOR2),
        ("greedy", 10, "10", GREEDY_COLOR3),
        ("greedy", 100, "100", GREEDY_COLOR4),
    ]

    dframes = [
        (get_data(kernelname, mode, beta), title, color)
        for mode, beta, title, color in names_and_colors
    ]

    for dframe in dframes:
        plot_regret_helper(dframe, axis, steps, individual)


def make_plot():
    """Plot the minmal regret for all combinations of kernels and lengthscales."""
    kernelname1 = "squaredexponential"
    fig, ax = plt.subplots(1, 2)
    # plot_regret("squaredexponential", "0.2", ax[0], 1000, individual=True)
    plot_regret("squaredexponential", "0.2", ax[0], 1000, individual=False)

    # plot_regret("matern", "0.2", ax[1], 1000, individual=True)
    plot_regret("matern", "0.2", ax[1], 1000, individual=False)

    # fig.suptitle("square exponential" + " 0.2")
    ax[0].set_title("square exponential")
    ax[1].set_title("Matern")
    handles, labels = ax[1].get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.1, 1.05))
    fig.supxlabel("number of function evaluations n")
    ax[0].set_ylabel("$\log \,min_n\, r_n$")

    plt.savefig(
        "./plots/beta_plot_TMLR.pdf",
        bbox_inches="tight",
    )
    plt.show()


make_plot()
