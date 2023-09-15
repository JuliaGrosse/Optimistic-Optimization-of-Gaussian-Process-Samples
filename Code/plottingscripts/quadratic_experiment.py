import numpy as np
import pandas as pd
import utils
import glob
import os
import parameters
import matplotlib.pyplot as plt
import plottingscripts.timing_results as time_plotting
import baselines.ucb_experiment as ucb_experiment
import GPOO.gpoo_experiment as gpoo_experiment

from tueplots import cycler
from tueplots.constants.color import palettes
from tueplots import figsizes, fonts, fontsizes


from tueplots import axes, bundles, figsizes

plt.rcParams.update(bundles.tmlr2023(nrows=1, ncols=2))
plt.rcParams.update(figsizes.tmlr2023(nrows=1, ncols=2))
# plt.rcParams.update(figsizes.icml2022_half())

###########################
# Definition of environment
###########################

GP_UCB_COLOR = "#F9521E"
GP_OO_COLOR = "#008D7C"
EI_COLOR = "#BF4684"

tue_gold = (174 / 255, 159 / 255, 109 / 255)
tue_darkgray = (55 / 255, 65 / 255, 74 / 255)
tue_lightgray = (175 / 255, 179 / 255, 183 / 255)
tue_lightgold = (239 / 255, 236 / 255, 226 / 255)
tue_red = (141 / 255, 45 / 255, 57 / 255)


KERNELPARAMS = parameters.Kernelparameters(
    kernelname=["polynomial"],
    variance=[1],
    lengthscale=[1],
    c=[0],
)
DOMAINPARAMS = parameters.Domainparameters(
    input_range=[(-1, 1), (-1, 1), (-1, 1)],
    nb_samples=20,
    step_size=100,
    discretization=10,
)
OPTIMIZERPARAMS = parameters.Optimizerparameters(
    steps=53,
    epsilon=0.05,
    init_ucb=3,
    beta=1,
    max_or_min="max",
    partition="black_box",
    mode="greedy",
    ucb_has_gradients=False,
)

EUCLIDEANOPTIMIZERPARAMS = parameters.Optimizerparameters(
    steps=53,
    epsilon=0.05,
    init_ucb=3,
    beta=1,
    max_or_min="max",
    partition="euclidean",
    mode="greedy",
)

PARAMS = parameters.Parameters(
    "polynomial_rebuttal_july3", KERNELPARAMS, DOMAINPARAMS, OPTIMIZERPARAMS
)


EUCLIDEANPARAMS = parameters.Parameters(
    "euclidean_polynomial_july3", KERNELPARAMS, DOMAINPARAMS, EUCLIDEANOPTIMIZERPARAMS
)


UCBEXP = ucb_experiment.UcbExperiment(PARAMS)
GPOOEXP = gpoo_experiment.GPOOExperiment(PARAMS)
EUCLIDEANGPOO = gpoo_experiment.GPOOExperiment(EUCLIDEANPARAMS)
###########################
# Run the search methods
###########################

gold = (174 / 255, 159 / 255, 109 / 255)
darkgray = (55 / 255, 65 / 255, 74 / 255)
lightgray = (175 / 255, 179 / 255, 183 / 255)
lightgold = (239 / 255, 236 / 255, 226 / 255)
red = (141 / 255, 45 / 255, 57 / 255)


def run_experiment():
    UCBEXP.generate_samples()
    UCBEXP.run_experiment()

    GPOOEXP.load_samples(
        filename="./results/groundtruth/['polynomial']/samples_polynomial.npy"
    )
    GPOOEXP.run_experiment()

    EUCLIDEANGPOO.load_samples(
        filename="./results/groundtruth/['polynomial']/samples_polynomial.npy"
    )
    EUCLIDEANGPOO.run_experiment()

    # post process logging info
    with open("./results/" + "/logging") as f:
        f_out = None
        for line in f:
            if "Start logging" in line:
                get_title = line.split("#")
                greedy = get_title[-1]
                expid = get_title[-2]
                kernelname = get_title[-3]
                title = utils.get_logging_filename(kernelname, greedy, expid)
                if f_out:
                    f_out.close()
                f_out = open(f"{title}.txt", "w")
            if f_out:
                f_out.write(line)
    if f_out:
        f_out.close()
    os.remove("./results/logging")


# run_experiment()
###########################
# Plot the regret
###########################
def get_data(experimentname, filename):
    """Load the results for the given kernel and lengthscale."""
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + "polynomial"
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def plot_regret_helper(dframe, axis, steps, color):
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

    # create list with standard deviation
    stds = []
    for i in range(steps):
        std = np.std([results[i] for results in min_simple_regret_list])
        stds.append(std)

    # plot the averade minimal regret
    axis.plot(
        range(len(average_min_regret)),
        np.asarray(average_min_regret),
        linestyle="-",
        linewidth=0.75,
        label=label,
        color=color,
    )

    axis.fill_between(
        range(len(average_min_regret)),
        np.asarray(average_min_regret - stds),
        np.asarray(average_min_regret + stds),
        stds,
        alpha=0.5,
        color=color,
    )


def plot_regret(axs):
    regret_ucb = get_data("polynomial", "_ucb_regret")
    regret_gpoo = get_data("polynomial_rebuttal_july3", "HOOregretgreedy")
    euclidean_regret_gpoo = get_data("euclidean_polynomial_july3", "HOOregretgreedy")

    plot_regret_helper((regret_ucb, "GP-UCB"), axs[0], 50, GP_UCB_COLOR)
    plot_regret_helper((regret_gpoo, "GP-OO (canonical)"), axs[0], 50, GP_OO_COLOR)
    plot_regret_helper(
        (euclidean_regret_gpoo, "GP-OO (euclidean)"), axs[0], 50, EI_COLOR
    )
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels)


########################
# Plot timing results
########################


def plot_times(axs):
    stored_times_gp = time_plotting.read_gp_ucb_times(
        "results/groundtruth/['polynomial']/loggingucb\n_polynomial.txt"
    )
    stored_times_hoo = time_plotting.read_hoo_times(
        "results/groundtruth/['polynomial']/HOOtimelogsgreedypolynomial_rebuttal_july2.txt",
        0,
    )

    ## Plot cumulative times for GP-UCB
    gp_acuisition = np.mean(stored_times_gp["acquisition"][:-1], axis=0)
    gp_updating = np.mean(stored_times_gp["updating"][:-1], axis=0)
    gp_evaluation = np.mean(stored_times_gp["evaluation"][:-1], axis=0)
    gp_iterations = np.mean(stored_times_gp["iterations"][:-1], axis=0)
    gp_overhead = gp_iterations - gp_acuisition - gp_updating - gp_evaluation

    gpucb_df = pd.DataFrame(
        {
            "overhead": gp_overhead[1:51],
            "evaluation": gp_evaluation[1:51],
            "acquisition": gp_acuisition[1:51],
            "GP updating": gp_updating[1:51],
        }
    )

    ## Plot times for HOO
    print(len(stored_times_hoo["iterations"]))
    hoo_iterations = np.mean(stored_times_hoo["iterations"], axis=0)
    hoo_construction = np.mean(stored_times_hoo["construction"], axis=0)
    hoo_evaluation = np.mean(stored_times_hoo["evaluation"], axis=0)
    hoo_overhead = hoo_iterations - hoo_construction - hoo_evaluation

    hoo_df = pd.DataFrame(
        {
            "overhead": np.repeat(hoo_overhead[1:], 2) / 2,
            "evaluation": np.repeat(hoo_evaluation[1:], 2) / 2,
            "construction": np.repeat(hoo_construction[1:], 2) / 2,
        }
    )

    axs[1] = gpucb_df.plot.area(
        ax=axs[1],
        color={
            "overhead": "#F9521E",
            "evaluation": tue_lightgold,
            "acquisition": "#BF4684",
            "GP updating": "black",
        },
    )

    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc="upper right")

    print(hoo_iterations, len(stored_times_hoo["iterations"][1]))
    axs[1].plot(np.repeat(hoo_iterations[:25], 2) / 2, color=GP_OO_COLOR)


fig, axs = plt.subplots(1, 2)
plot_regret(axs)
plot_times(axs)
plt.tight_layout()
axs[0].set_ylabel("min regret")
axs[1].set_ylabel("time in s")
axs[0].set_xlabel("iteration")
axs[1].set_xlabel("iteration")

plt.savefig(
    "./plots/quadratic_kernel_experiment_TMLR_rebuttal.pdf", bbox_inches="tight"
)
plt.show()
