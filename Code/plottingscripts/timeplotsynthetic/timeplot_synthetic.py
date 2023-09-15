"""Figure with the grid search results for GP-UCB."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import cycler
from tueplots.constants.color import palettes
from tueplots import figsizes, fonts, fontsizes
import tikzplotlib as plt2tikz

from plottingscripts.timing_results import (
    plot_regret_helper_refactored as plot_regret_helper,
)
from plottingscripts.timing_results import final_cumulative_time_plot, _subtract_times
from utils import get_function_evaluation_data as my_get_data

import matplotlib


from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=1, ncols=2))
plt.rcParams.update(figsizes.tmlr2023(nrows=1, ncols=2))
# plt.rcParams.update(figsizes.icml2022_half())


def plot_regret(kernelname, lengthscale, axis, steps):
    """Plot the minmal regret for the specified kernel and lengthscale on the given axis."""
    names_and_colors = [
        ("HOOregretdiscretization", "GP-00", 1),
        ("ucb_regret", "GP-UCB", 1),
        ("randomregret", "random", 1),
    ]
    print("Plot regret is called w ith", kernelname, lengthscale, axis, steps)
    dframes = [
        (my_get_data(kernelname, name, discretization), title)
        for name, title, discretization in names_and_colors
    ]

    for dframe in dframes:
        plot_regret_helper(dframe, axis, steps, individual=True)


"""Figure with timing results on synthetic functions for Squared Exponential and Matern."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import cycler
from tueplots.constants.color import palettes
from tueplots import figsizes, fonts, fontsizes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


matplotlib_default_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


Evaluation_costs = 0

tue_gold = (174 / 255, 159 / 255, 109 / 255)
tue_darkgray = (55 / 255, 65 / 255, 74 / 255)
tue_lightgray = (175 / 255, 179 / 255, 183 / 255)
tue_lightgold = (239 / 255, 236 / 255, 226 / 255)
tue_red = (141 / 255, 45 / 255, 57 / 255)

GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"


def _store_times(stored_times):
    """Helper function to read the timing results of GP-UCB."""
    iteration_times = _subtract_times(stored_times, "iterations", "iterations2")
    updating_times = _subtract_times(stored_times, "updating", "acquisition")
    acquisition_times = _subtract_times(stored_times, "acquisition", "evaluation")
    evaluation_times = _subtract_times(stored_times, "evaluation", "iterations2")
    iteration_times = [
        iteration_time + Evaluation_costs for iteration_time in iteration_times
    ]

    stored_times["iterations"][-1] = iteration_times
    stored_times["updating"][-1] = updating_times
    stored_times["acquisition"][-1] = acquisition_times
    stored_times["evaluation"][-1] = evaluation_times


def _new_sample(stored_times):
    """Helper function to read the timing results of GP-UCB."""
    for keyword in stored_times.keys():
        old = stored_times[keyword]
        old.append([])
        stored_times[keyword] = old
    return stored_times


def read_gp_ucb_times(filename):
    """Collect the times GP-UCB needed for each iteration, the updating of the GP-Posterior,
    the optimization of the acquisition function and the evaluation of the function
    from the logging file."""
    print("open", filename)
    with open(filename) as timelog_file:

        stored_times = {
            "iterations": [[]],
            "updating": [[]],
            "acquisition": [[]],
            "evaluation": [[]],
        }

        skip = False
        for line in timelog_file:
            words = line.split()
            time = int(words[0]) * 0.001  # milliseconds to seconds
            if "Iteration" in line:
                stored_times["iterations"][-1].append(time)
            if "Updating parameters of the model" in line:
                if not skip:
                    stored_times["updating"][-1].append(time)
                else:
                    skip = False
            if "Starting gradient-based optimization" in line:
                stored_times["acquisition"][-1].append(time)
            if "Evaluating user function" in line:
                stored_times["evaluation"][-1].append(time)
            if "Stopped after" in line:
                stored_times["iterations"][-1].append(time)
                skip = True

                _store_times(stored_times)
                _new_sample(stored_times)
    return stored_times


def read_turbo_times(filename, costs=0, batchsize=10, nb_init=25):
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
    return stored_times


def read_turbo_evals(filename, costs=0, batchsize=10, nb_init=25):
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            line = line[2:]
            line = line[:-1]
            words = line.split("#")
            for i, word in enumerate(words[1:]):
                iteration_time = float(word) + costs
                iteration_times.append(iteration_time)
            all_iteration_times.append(iteration_times)
    stored_times["evals"] = all_iteration_times
    return stored_times


def read_hoo_times(filename):
    """Collect the times hoo needed for each iteration, the construction part of an iteration
    and the evaluation part of an iteration from the logging file."""
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times, all_construction_times, all_evaluation_times = [], [], []
        for line in timelog_file:
            iteration_times, construction_times, evaluation_times = [], [], []
            words = line.split("#")
            for i, word in enumerate(words):
                if i % 3 == 0:
                    iteration_times.append(float(word) + 0)
                if i % 3 == 1:
                    construction_times.append(float(word))
                if i % 3 == 2:
                    evaluation_times.append(float(word) + 0)
            all_iteration_times.append(iteration_times)
            all_construction_times.append(construction_times)
            all_evaluation_times.append(evaluation_times)
    stored_times["iterations"] = all_iteration_times
    stored_times["construction"] = all_construction_times
    stored_times["evaluation"] = all_evaluation_times
    return stored_times


def read_random_times(filename):
    """Collect the times hoo needed for each iteration, the construction part of an iteration
    and the evaluation part of an iteration from the logging file."""
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            words = line.split("#")
            for i, word in enumerate(words):
                iteration_times.append(float(word) + Evaluation_costs)
            all_iteration_times.append(iteration_times)
    stored_times["iterations"] = all_iteration_times
    return stored_times


def plot_cumulative_times(
    filename_gp, filename_hoo, filename_turbo_times, filename_turbo_evals
):
    """Plot the time per iteration for the two methods."""

    fig, axs = plt.subplots(1, 2)

    ### Plotting for GP UCB
    stored_times_gp = read_gp_ucb_times(filename_gp)
    gp_acuisition = np.mean(stored_times_gp["acquisition"][:-1], axis=0)
    gp_updating = np.mean(stored_times_gp["updating"][:-1], axis=0)
    gp_evaluation = np.mean(stored_times_gp["evaluation"][:-1], axis=0)
    gp_iterations = np.mean(stored_times_gp["iterations"][:-1], axis=0)
    gp_overhead = gp_iterations - gp_acuisition - gp_updating - gp_evaluation

    gpucb_df = pd.DataFrame(
        {
            "overhead": gp_overhead[1:],
            "evaluation": gp_evaluation[1:],
            "acquisition": gp_acuisition[1:],
            "GP updating": gp_updating[1:],
        }
    )
    gpucb_df.plot.area(
        ax=axs[0],
        color={
            "overhead": "#F9521E",
            "evaluation": tue_lightgold,
            "acquisition": "#008D7C",
            "GP updating": "black",
        },
    )

    ### Plotting for GP OO
    stored_times_hoo = read_hoo_times(filename_hoo)
    hoo_iterations = np.mean(stored_times_hoo["iterations"], axis=0)
    repeated_hoo_iterations = np.repeat(hoo_iterations, 2) / 2
    axs[1].plot(repeated_hoo_iterations, color=GP_OO_COLOR, label="GP-OO")

    ### Plotting for TurBO
    stored_times_turbo = read_turbo_times(filename_turbo_times)
    stored_evals_turbo = read_turbo_evals(filename_turbo_evals)
    turbo_iterations = stored_times_turbo["iterations"]
    turbo_evals = stored_evals_turbo["evals"]
    turbo_all_times = []
    for i in range(20):
        times = []
        prev_evals = 0
        turbo_evals[i] = [25] + turbo_evals[i]
        for j in range(len(turbo_iterations[i])):
            nb_evals = turbo_evals[i][j] - prev_evals
            average_time = turbo_iterations[i][j] / nb_evals
            prev_evals = turbo_evals[i][j]
            times += [average_time] * int(nb_evals)
        turbo_all_times.append(times[:1000])

    axs[1].plot(
        np.mean(np.stack(turbo_all_times), axis=0), color=TURBO_COLOR, label="TurBO"
    )
    axs[1].legend()

    # axs[1].set_ylabel("time in seconds")
    axs[1].set_xlabel("iteration")

    axs[0].set_ylabel("time in seconds")
    axs[0].set_xlabel("iteration")

    plt2tikz.save("./plottingscripts/timeplotsynthetic/timing_plot_synthetic.tex")
    plt.savefig(
        "./plottingscripts/timeplotsynthetic/timing_plot_synthetic_rebuttal.pdf",
        bbox_inches="tight",
    )
    plt.show()


def get_data(filename, kernelname, benchmark="groundtruth"):
    pathname = "./results/" + benchmark + "/['" + kernelname + "']/" + filename
    with open(pathname, "r") as temp_f:
        col_count = [len(l.split("#")) for l in temp_f.readlines()]
        column_names = list(range(0, max(col_count)))
    regret_df = pd.read_csv(pathname, sep="#", header=None, names=column_names)
    return regret_df


plot_cumulative_times(
    "./plottingscripts/timeplotsynthetic/timeplot_results/ucb_logs.txt",
    "./plottingscripts/timeplotsynthetic/timeplot_results/HOOtimelogsgreedythreedimse.txt",
    "./plottingscripts/timeplotsynthetic/timeplot_results/turbotimelogsturbothreedimseturbo.txt",
    "./plottingscripts/timeplotsynthetic/timeplot_results/turboevalsthreedimseturbo.txt",
)
