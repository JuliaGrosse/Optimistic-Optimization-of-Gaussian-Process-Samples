"""Some useful functions."""
import os
import math
import itertools
from glob import glob
import numpy as np
import benchmark_functions
import random
import pandas as pd

random.seed(28)


def number_of_zero_crossings(u, kernelname, l):
    if kernelname == "squaredexponential":
        ktwo0 = -l / l ** 2
    elif kernelname == "matern":
        ktwo0 = -3 * l / l ** 3
    else:
        raise NotImplementedError()
    Nu = np.sqrt(-ktwo0) * np.exp(-(u ** 2) / 2) / (2 * math.pi)
    return Nu


def upper_bound(params):
    x = np.sum(
        [
            number_of_zero_crossings(x, params.kernelname[0], params.lengthscale[0])
            for x in np.linspace(0, 10, 100)
        ]
    )
    return x


def make_nx_node(node, nodename):
    """Store the observed node in the networkx format.

    :param (Float, [Float], Floar, ?, Int) node: Observed node.
    :return: Networkx node.
    :rtype: (String, {String:?})

    """
    uti, root_cut, obs, intervallenghts, depth = node
    return (
        nodename,
        {
            "utility": uti,
            "cut": root_cut,
            "observation": obs,
            "intervallenghts": intervallenghts,
            "depth": depth,
        },
    )


def open_log_files(experiment, names):
    """Open files to log the results.

    :param Experiment experiment: Experiment
    :param [String] names: Names of the logging files
    :return: Dictionary with result files.
    :rtype: {String:File}

    """
    files = {}
    for name in names:
        file = open(get_filename(experiment, name) + ".txt", "a+")
        files[name] = file
    return files


def write_logs(obs_file, observations):
    """Log the observed points.

    :param obs_file: txt file
    :param [Float] observations: observed points
    :return: None
    :rtype: None

    """
    obs_string = ""
    for obs in observations:
        obs_string += str(obs)
        obs_string += "#"
    obs_string = obs_string[:-1]
    obs_string += "\n"
    obs_file.write(obs_string)


def write_timelogs(timelogs_file, timelogs):
    """Log the passed time per iteration.

    :param obs_file: txt file
    :param [Float] observations: observed points
    :return: None
    :rtype: None

    """
    obs_string = ""
    for obs in timelogs:
        for obs_i in obs:
            obs_string += str(obs_i)
            obs_string += "#"
    obs_string = obs_string[:-1]
    obs_string += "\n"
    timelogs_file.write(obs_string)


def write_timelogs_turbo(timelogs_file, timelogs):
    timelogs_file.write(str(timelogs) + "\n")


def write_timelogs_adabkb(timelogs_file, timelogs):
    timelogs_file.write(str(timelogs) + "\n")


def write_timelogs_direct(timelogs_file, timelogs):
    timelogs_file.write(str(timelogs) + "\n")


def get_filename(experiment, name):
    """Define path for logging file.

    :param Experiment experiment: Experiment.
    :param String name: Name of the logging file.
    :return: Filepath
    :rtype: String

    """

    filename = (
        "./results/"
        + str(experiment.benchmark)
        + "/"
        + str(experiment.params.kernelname)
        + "/"
        + name
        + str(experiment.params.ID)
    )
    return filename


def _corner_points(params, i):
    """Helper function for sampling a high dimensional polynom.

    :param Parameters params: Experimental configuration.
    :param Int i: Index of the kernel function.
    :return: Corner points, Sampled values for corner points
    :rtype: np.array, np.array

    """

    def cov(x1, x2):
        return params.variance[i] * (np.dot(x1, x2) + params.c[i]) ** 2

    corner_points = itertools.product(*params.input_range)
    corner_points = [np.asarray(corner_point) for corner_point in corner_points]
    gram = np.asarray([[cov(x1, x2) for x2 in corner_points] for x1 in corner_points])
    mean = np.zeros((len(corner_points)))
    samples = np.random.multivariate_normal(mean, gram, params.nb_samples)
    return corner_points, samples


def _linear_equation_system(params, i):
    """Helper function for sampling a high dimensional polynom.

    :param Parameters params: Experimental configuration.
    :param Int i: Index of the kernel function.
    :return: Coefficients for the samples polynom.
    :rtype: [np.Float]

    """
    corner_ps, samples = _corner_points(params, i)
    equations = np.stack(
        [
            [
                cp[i] * cp[j]
                for i in range(params.dim)
                for j in range(params.dim)
                if j >= i
            ]
            for cp in corner_ps
        ]
    )
    coefficients = [
        np.linalg.lstsq(equations, samples[i]) for i in range(params.nb_samples)
    ]
    return coefficients


def _calculate_polynom(params, coefficients):
    """Helper function for sampling a high dimensional polynom.

    :param Parameters params: Experimental configuration.
    :param Int i: Index of the kernel function.
    :return: Coefficients for the samples polynom.
    :rtype: [np.Float]

    """
    domain = np.meshgrid(
        *[
            np.arange(start, stop, params.range_length / params.step_size)
            for start, stop in params.input_range
        ],
        indexing="ij",
    )
    non_coeffs = [
        domain[i] * domain[j]
        for i in range(params.dim)
        for j in range(params.dim)
        if j >= i
    ]
    res = np.sum(
        [non_coeffs[i] * coefficients[0][i] for i in range(len(coefficients[0]))],
        axis=0,
    )
    return res


def polynomial_samples(params, j):
    """Sampling of a high-dimensional polynom.

    :param Parameters params: Experimental configuration.
    :param Int i: Index of the kernel function.
    :return: Sampled functions
    :rtype:  np.array[Float]

    """
    coeefs = _linear_equation_system(params, j)
    samples = np.asarray(
        [_calculate_polynom(params, coeefs[i]) for i in range(params.nb_samples)]
    )
    return samples


def remove_experiment(ID, remove_samples=False):
    """Delete logging files for an experiment. Be careful: ID should not contain "samples"
    as substring ...

    :param String ID: name of the experiment.
    :param Boolean remove_samples: Do you really want to remove the sampled functions?
    :return: None
    :rtype: None

    """
    for file in glob("./results/*/*/*" + str(ID) + "*"):
        if not remove_samples:
            if not "samples" in file:
                os.remove(file)
        else:
            os.remove(file)


def clear_folder(folderpath):
    """Remove all files in the specified folder."""
    files = glob(folderpath)
    for f in files:
        os.remove(f)


def get_logging_filename(kernelname, greedy, expid):
    filename = (
        "./results/"
        + "groundtruth"
        + "/"
        + kernelname
        + "/"
        + "/logging"
        + greedy
        + "_"
        + expid
    )
    return filename


def get_logging_filename_benchmarks(benchmark, kernelname, greedy, expid):
    filename = (
        "./results/"
        + benchmark
        + "/"
        + kernelname
        + "/"
        + "/logging"
        + greedy
        + "_"
        + expid
    )
    return filename


def get_function_evaluation_data(kernelname, filename, discretization):
    """Load the results for the given kernel and lengthscale."""
    regret_dframe = pd.read_csv(
        "./results/"
        + "groundtruth"
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + kernelname
        + "functionevaluationexperiment"
        + "_discretizationsize_"
        + str(discretization)
        + ".txt",
        sep="#",
        header=None,
    )
    return regret_dframe


def basic_get_filename(kernelname, greedy, expid):
    filename = (
        "./results/"
        + "groundtruth"
        + "/"
        + kernelname
        + "/"
        + "/logging"
        + greedy
        + "_"
        + expid
    )
    return filename


def post_process_logging_info(get_filename):
    # post process logging info
    with open("./results/" + "/logging") as f:
        f_out = None
        for line in f:
            if "Start logging" in line:
                get_title = line.split("#")
                greedy = get_title[-1]
                expid = get_title[-2]
                kernelname = get_title[-3]
                title = get_filename(kernelname, greedy, expid)
                if f_out:
                    f_out.close()
                f_out = open(f"{title}.txt", "w")
                print("Done with logging.", line)
            if f_out:
                f_out.write(line)
    if f_out:
        f_out.close()


def sample_domain(benchmark, domain):
    new_domain = []
    minimum_location = benchmark_functions.MINIMA[benchmark][0]
    for dimension in range(len(minimum_location)):
        lower = random.uniform(domain[dimension][0], minimum_location[dimension])
        upper = random.uniform(minimum_location[dimension], domain[dimension][1])
        new_domain.append([lower, upper])
    return new_domain
