"""Definition of diameters for GP-OO."""
import itertools
import sys
import numpy as np

sys.path.append(".")
from . import construct_children_utils
import kernel_functions


def get_gpoo_diameter_term(params):
    """Select the right diameter function for the specified kernel.

    :param Parameters params: Experimental configuration.
    :return: Function to calculate the diameters.
    :rtype: Function

    """
    diameters = []
    for i, name in enumerate(params.kernelname):
        if name in ["matern", "squaredexponential", "linear"]:
            diam = gpoo_corner_diameters(params, i)
        else:
            diam = gpoo_black_box_diameters(params, i)
        diameters.append(diam)
    return sum_diameters(diameters)


def sum_diameters(diam_functions):
    """Sum of Daimeters.

    :param [functions] diam_functions: Diameter functions.
    :return: Sum diameter function.
    :rtype: Function.

    """

    def sum_diameter(ilenghts, centerpoint):
        return np.sum([diameter(ilenghts, centerpoint) for diameter in diam_functions])

    return sum_diameter


def gpoo_corner_diameters(params, i):
    """Diameters for the metric induced by euclidean-like metrics.

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of diameter function.
    :return: Function to calculate the diameters.
    :rtype: Function

    """
    kernelfunction = kernel_functions.get_my_kernel_by_name(params, i)
    distance_function = construct_children_utils.distance(kernelfunction)

    def diameters(ilengths, centerpoint):
        half_lengths = [(ilength[1] - ilength[0]) / 2 for ilength in ilengths]
        return distance_function(np.zeros(len(centerpoint)), np.asarray(half_lengths))

    return diameters


def gpoo_black_box_diameters(params, i):
    """Diameters for the metric induced by an arbitrary kernel.

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of diameter function.
    :param String greedy: How to set beta.
    :return: Function to calculate the diameters.
    :rtype: Function

    """
    kernelfunction = kernel_functions.get_my_kernel_by_name(params, i)
    distance_function = construct_children_utils.distance(kernelfunction)

    def diameters(ilengths, centerpoint):
        linspaces = [
            list(np.linspace(ilength[0], ilength[1], 10)) for ilength in ilengths
        ]
        distances = []
        center = np.asarray(centerpoint)
        for other_point in itertools.product(*linspaces):
            distance_i = distance_function(center, np.asarray(other_point))
            distances.append(distance_i)
        return np.max(distances)

    return diameters
