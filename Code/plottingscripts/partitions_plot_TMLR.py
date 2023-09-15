"""Plot with four examples of runs with GP-OO and GP-UCB on Matern, Squared Exponential and
Qudratic kernel."""
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from tueplots.constants.color import rgb
from tueplots import cycler
from tueplots.constants.color import palettes
from tueplots import figsizes, fonts, fontsizes

import parameters
import baselines.ucb_experiment as ucb_experiment
import GPOO.gpoo_experiment as gpoo_experiment

####################################################################################################
# Attention:
# To make these plots the two commented lines in GPOO.optimize.TreeSearchOptimizer.optimize()
# need to be uncommented.
####################################################################################################


# specific the layout of the plots to match Neurips2022 requirements
from tueplots import axes, bundles, figsizes

bundles.tmlr2023()

plt.rcParams["axes.linewidth"] = 0.2
plt.rcParams.update(bundles.tmlr2023(nrows=2, ncols=4))
plt.rcParams.update(figsizes.tmlr2023(nrows=2, ncols=4, height_to_width_ratio=1))


random.seed(28)

GP_UCB_COLOR = "#F9521E"
EI_COLOR = "#BF4684"
TURBO_COLOR = "#CBAE11"
GP_OO_COLOR = "#008D7C"
####################################################################################################
# Plotting Functions
####################################################################################################
def define_patches(heap):
    """Build patches from the nodes that are stored in the heap."""
    patches, values = [], []
    for name, node in heap.nodes(data=True):
        uti, obs, ilen, = (
            node["utility"],
            node["observation"],
            node["intervallenghts"],
        )
        if heap.out_degree(name) == 0:
            rectangle = Rectangle(
                (ilen[0][0], ilen[1][0]),
                ilen[0][1] - ilen[0][0],
                ilen[1][1] - ilen[1][0],
                lw=0.2,
                edgecolor="black",  # rgb.tue_gold,
            )
            patches.append(rectangle)
            values.append(uti - obs)
            print(uti, obs)
    print("define patches", len(values))
    return patches, np.asarray(values)


def plot_partition(seexp, sample_index, ax):
    """Plot the parition that corresponds to the GP-OO run."""
    # ax.contour(seexp.domain[0], seexp.domain[1], seexp.samples[-sample_index], cmap="bone", vmin=0, vmax=12)
    for point, _, _ in seexp.gpoo_stored_nodes[-sample_index]:
        ax.plot(
            point[0], point[1], "x", markersize=2, color=GP_OO_COLOR
        )  # rgb.tue_gold)
    patches, values = define_patches(seexp.gpoo_stored_maxheaps[-sample_index][1])
    coll = matplotlib.collections.PatchCollection(
        patches, cmap="gray_r", match_original=True
    )
    coll.set_array(values)
    coll.set_clim([0, np.max(values)])
    rectangles = ax.add_collection(coll)
    return rectangles


def calc_cut_point(seexp, cut_point, dim):
    """Helper function for plotting evaluation locations."""
    total_distance = seexp.params.input_range[dim][1] - seexp.params.input_range[dim][0]
    cut_distance = cut_point - seexp.params.input_range[dim][0]
    distance = cut_distance / total_distance
    cut_index = np.min(
        [
            np.max([0, int(distance * seexp.samples[0].shape[dim])]),
            seexp.samples[0].shape[dim] - 1,
        ]
    )
    return (
        seexp.params.input_range[dim][0]
        + (cut_index / seexp.samples[0].shape[dim]) * total_distance
    )


def plot_contour(seexp, sample_index, ax):
    """Plot contour lines for the functions and mark the locations where GP-OO and GP-UCB
    evaluated the function."""
    maxp = np.unravel_index(
        seexp.samples[-sample_index].argmax(), seexp.samples[-sample_index].shape
    )
    x = seexp.domain[0][maxp[0]][0]
    y = seexp.domain[1][maxp[1]][maxp[1]]
    ax.plot(x, y, "*", markersize=3, color=rgb.tue_gold, alpha=1)
    con = ax.contourf(
        seexp.domain[0],
        seexp.domain[1],
        seexp.samples[-sample_index],
        cmap="Greys_r",
        levels=30,
        alpha=0.85,
    )
    for point, utility, _ in seexp.gpoo_stored_nodes[-sample_index]:
        ax.plot(point[0], point[1], "x", markersize=2, color=GP_OO_COLOR)
    ax.plot(point[0], point[1], "x", markersize=2, color=GP_OO_COLOR, label="GP-OO")
    maxp = np.unravel_index(
        seexp.samples[-sample_index].argmax(), seexp.samples[-sample_index].shape
    )
    x = seexp.domain[0][maxp[0]][0]
    y = seexp.domain[1][maxp[1]][maxp[1]]
    ax.plot(x, y, "*", markersize=3, color=rgb.tue_gold, alpha=1)

    return con


def plot_contour_ucb(seexp, sample_index, ax):
    """Plot contour lines for the functions and mark the locations where GP-OO and GP-UCB
    evaluated the function."""
    maxp = np.unravel_index(
        seexp.samples[-sample_index].argmax(), seexp.samples[-sample_index].shape
    )
    x = seexp.domain[0][maxp[0]][0]
    y = seexp.domain[1][maxp[1]][maxp[1]]
    con = ax.contourf(
        seexp.domain[0],
        seexp.domain[1],
        seexp.samples[-sample_index],
        cmap="Greys_r",
        levels=30,
        alpha=0.85,
    )
    for point1, point2, _ in seexp.gp_ucb_observations[sample_index]:
        point1 = calc_cut_point(seexp, point1, 0)
        point2 = calc_cut_point(seexp, point2, 1)
        ax.plot(point1, point2, "x", markersize=2, color=GP_UCB_COLOR, clip_on=True)
    ax.plot(point1, point2, "x", markersize=2, color=GP_UCB_COLOR, label="GP-UCB")
    maxp = np.unravel_index(
        seexp.samples[-sample_index].argmax(), seexp.samples[-sample_index].shape
    )
    x = seexp.domain[0][maxp[0]][0]
    y = seexp.domain[1][maxp[1]][maxp[1]]
    ax.plot(x, y, "*", markersize=3, color=rgb.tue_gold, alpha=1)

    return con


####################################################################################################
# Initialize Figure
####################################################################################################
FIG, AXS = plt.subplots(2, 4)

OPTIMIZERPARAMS1 = parameters.Optimizerparameters(
    steps=40,
    epsilon=0.05,
    init_ucb=5,
    beta=None,
    max_or_min="max",
    partition="euclidean",
    ucb_discretization=1,
    mode="heuristic",
)

OPTIMIZERPARAMS11 = parameters.Optimizerparameters(
    steps=98,
    epsilon=0.05,
    init_ucb=5,
    beta=None,
    max_or_min="max",
    partition="euclidean",
    ucb_discretization=1,
    mode="heuristic",
)

DOMAINPARAMS1 = parameters.Domainparameters(
    input_range=[(0, 5), (0, 5)],
    nb_samples=2,
    step_size=100,
    discretization=1,
)

KERNELPARAMS1 = parameters.Kernelparameters(
    kernelname=["polynomial"],
    variance=[1],
    lengthscale=[1],
    c=[0],
)

DOMAINPARAMS2 = parameters.Domainparameters(
    input_range=[(-1, 1), (-1, 1)],
    nb_samples=1,
    step_size=100,
    discretization=5,
)

####################################################################################################
# Polynomial: Optimized Partitioning
####################################################################################################
def add_polynomial_optimal(axs):
    """Run GP-OO and GP-UCB on a sampled quadratic function with optimized partitioning."""
    column = 3

    optimizerparams = parameters.Optimizerparameters(
        steps=15,
        epsilon=0.0005,
        init_ucb=5,
        beta=1,
        max_or_min="max",
        partition="black_box",
        ucb_has_gradients=False,
        mode="greedy",
    )
    params = parameters.Parameters(
        "partitions_polynomial_black_box", KERNELPARAMS1, DOMAINPARAMS2, optimizerparams
    )

    ucbexp = ucb_experiment.UcbExperiment(params)
    gpooexp = gpoo_experiment.GPOOExperiment(params)
    ucbexp.generate_samples()
    ucbexp.run_experiment()
    samples = ucbexp.samples.copy()
    gpooexp.samples = samples
    gpooexp.run_experiment()

    stored_polynomial_samples = gpooexp.samples.copy()

    _ = plot_partition(gpooexp, 0, axs[1, column])
    _ = plot_contour(gpooexp, 0, axs[0, column])
    _ = plot_contour_ucb(ucbexp, 0, axs[0, column])
    axs[1, column].set_xlim([-1, 1])
    axs[1, column].set_ylim([-1, 1])
    axs[0, column].set_xlim([-1, 1])
    axs[0, column].set_ylim([-1, 1])

    axs[1, column].set_xticks([])
    axs[1, column].set_yticks([])
    axs[0, column].set_xticks([])
    axs[0, column].set_yticks([])
    return stored_polynomial_samples


####################################################################################################
# Polynomial: Euclidean Partitioning
####################################################################################################


def add_polynomial_nonoptimal(axs, samples):
    """Run GP-OO and GP-UCB on a sampled quadratic function with regular paritioning."""
    optimizerparams = parameters.Optimizerparameters(
        steps=15,
        epsilon=0.0005,
        init_ucb=5,
        beta=1,
        max_or_min="max",
        partition="euclidean",
        ucb_has_gradients=False,
        mode="greedy",
    )
    params = parameters.Parameters(
        "partitions_polynomial_euclidean", KERNELPARAMS1, DOMAINPARAMS2, optimizerparams
    )

    column = 2

    ucbexp = ucb_experiment.UcbExperiment(params)
    gpooexp = gpoo_experiment.GPOOExperiment(params)
    ucbexp.samples = samples
    gpooexp.samples = samples
    ucbexp.run_experiment()
    gpooexp.run_experiment()

    _ = plot_partition(gpooexp, 0, axs[1, column])
    _ = plot_contour(gpooexp, 0, axs[0, column])
    _ = plot_contour_ucb(ucbexp, 0, axs[0, column])
    axs[1, column].set_xlim([-1, 1])
    axs[1, column].set_ylim([-1, 1])
    axs[0, column].set_xlim([-1, 1])
    axs[0, column].set_ylim([-1, 1])

    axs[1, column].set_xticks([])
    axs[1, column].set_yticks([])
    axs[0, column].set_xticks([])
    axs[0, column].set_yticks([])


####################################################################################################
# Squared Exponential
####################################################################################################
def add_se(axs):
    """Run GP-OO and GP-UCB on a sampled from GP with square exponential kernel."""
    kernelparams = parameters.Kernelparameters(
        kernelname=["squaredexponential"],
        variance=[1],
        lengthscale=[1],
        c=[0],
    )
    params = parameters.Parameters(
        "partitions_squaredexponential", kernelparams, DOMAINPARAMS1, OPTIMIZERPARAMS1
    )
    column = 0

    ucbexp = ucb_experiment.UcbExperiment(params)
    gpooexp = gpoo_experiment.GPOOExperiment(params)
    ucbexp.generate_samples()
    ucbexp.run_experiment()
    samples = ucbexp.samples.copy()
    gpooexp.samples = samples
    gpooexp.run_experiment()

    _ = plot_partition(gpooexp, 0, axs[1, column])
    _ = plot_contour(gpooexp, 0, axs[0, column])
    _ = plot_contour_ucb(ucbexp, 0, axs[0, column])
    axs[1, column].set_xlim([0, 5])
    axs[1, column].set_ylim([0, 5])
    axs[0, column].set_xlim([0, 5])
    axs[0, column].set_ylim([0, 5])

    axs[1, column].set_xticks([])
    axs[1, column].set_yticks([])
    axs[0, column].set_xticks([])
    axs[0, column].set_yticks([])


####################################################################################################
#  Matern
####################################################################################################
def add_matern(axs):
    """Run GP-OO and GP-UCB on a sampled from GP with Matern kernel."""
    kernelparams = parameters.Kernelparameters(
        kernelname=["matern"],
        variance=[1],
        lengthscale=[1],
        c=[0],
    )

    column = 1

    params = parameters.Parameters(
        "partitions_matern", kernelparams, DOMAINPARAMS1, OPTIMIZERPARAMS11
    )

    ucbexp = ucb_experiment.UcbExperiment(params)
    gpooexp = gpoo_experiment.GPOOExperiment(params)
    ucbexp.generate_samples()
    ucbexp.run_experiment()
    samples = ucbexp.samples.copy()
    gpooexp.samples = samples
    gpooexp.run_experiment()

    rectangles = plot_partition(gpooexp, 0, axs[1, column])
    con = plot_contour(gpooexp, 0, axs[0, column])
    _ = plot_contour_ucb(ucbexp, 0, axs[0, column])

    return rectangles, con


def add_layout(axs):
    """Specify layout of plot."""
    axs[1, 0].set_xlim([0, 5])
    axs[1, 0].set_ylim([0, 5])
    axs[1, 1].set_xlim([0, 5])
    axs[1, 1].set_ylim([0, 5])

    axs[1, 0].set_xbound(lower=0, upper=5)
    axs[1, 0].set_ybound(lower=0, upper=5)
    axs[1, 1].set_xbound(lower=0, upper=5)
    axs[1, 1].set_ybound(lower=0, upper=5)

    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    axs[0, 0].set_title("square exponential")
    axs[0, 1].set_title("Matérn 3/2")
    axs[0, 3].set_title("quadratic (canonical)")
    axs[0, 2].set_title("quadratic (regular)")


SAMPLES = add_polynomial_optimal(AXS)
add_polynomial_nonoptimal(AXS, SAMPLES)
add_se(AXS)
add_layout(AXS)
_, CON = add_matern(AXS)
handles, labels = AXS[0, 1].get_legend_handles_labels()
AXS[0, 3].legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.savefig("./plots/partitions_TMLR_rebuttal.pdf")
plt.show()
