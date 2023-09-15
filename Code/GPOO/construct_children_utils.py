"""Search space partitioning."""

import sys

sys.path.append(".")

import random
import numpy as np
import itertools

import kernel_functions

np.random.seed(28)
random.seed(28)


def construct_root_node(params):
    """Calculate the cut_point for the root node.

    :param Parameters params: Experimental configuration.
    :return: Cutpoints and intervallengths for each dimension.
    :rtype: [Float], [[Float]]

    """
    if params.partition == "euclidean":
        return euclidean_root(params)
    return black_box_root(params)


def black_box_root(params):
    """Calculate the cut_point for the root node based on a black box metric.

    :param Parameters params: Experimental configuration.
    :return: Coordindates of cutput, Coordindates of upper and lower bounds
                     for each dimension.
    :rtype: [Floats], [(Float, Float)]

    """
    center, _, _ = get_center(params.input_range, params, 0)
    return center, params.input_range


def euclidean_root(params):
    """Calculate the cut_point for the root node based on the euclidean metric.

    :param Parameters params: Experimental configuration.
    :return: Coordindates for each dimension, Coordindates of upper and lower bound
                     for each dimension.
    :rtype: [Floats], [(Float, Float)]

    """

    def interval(i):
        return (params.input_range[i][0], params.input_range[i][1])

    def intervallength(i):
        return params.input_range[i][1] - params.input_range[i][0]

    def cutpoint(i):
        return params.input_range[i][0] + intervallength(i) / 2

    intervalls = [interval(i) for i in range(len(params.input_range))]
    cuts = [cutpoint(i) for i in range(len(params.input_range))]
    return cuts, intervalls


def euclidean_metric(cutpoints, intervals, params):
    """Construct cutpoints and intervallengths for the children based on the
    Euclidean metric.

    :param [Float] cutpoints: Cut points of parent node for each dimension.
    :param [(Float, Float)] intervals: Interval bounds of parent node for each dimension.
    :return: Cutpoints and interval bounds for left and right child.
    :rtype: [Float], [(Float, Float)], [Float], [(Float, Float)]

    """
    if params.nb_children == 2:
        intervallengths = [u - l for l, u in intervals]
        maxindices = np.argwhere(intervallengths == np.amax(intervallengths))
        if len(maxindices) > 1:
            np.random.seed(28)
            argmax_dimension = np.random.choice(np.squeeze(maxindices))
        else:
            argmax_dimension = maxindices[0][0]
        new_length = np.asarray(intervallengths)[argmax_dimension] * 0.5
        left_cut, right_cut = np.asarray(cutpoints, dtype=float), np.asarray(
            cutpoints, dtype=float
        )
        left_i, right_i = intervals.copy(), intervals.copy()
        left_cut[argmax_dimension] = cutpoints[argmax_dimension] - new_length * 0.5
        right_cut[argmax_dimension] = cutpoints[argmax_dimension] + new_length * 0.5
        left_i[argmax_dimension] = (
            intervals[argmax_dimension][0],
            cutpoints[argmax_dimension],
        )
        right_i[argmax_dimension] = (
            cutpoints[argmax_dimension],
            intervals[argmax_dimension][1],
        )
        return [list(left_cut), list(right_cut)], [list(left_i), list(right_i)]
    else:
        intervalllengths = [u - l for l, u in intervals]
        cuts_per_dim = params.nb_children
        children_lengths = [
            ils / cuts_per_dim for ils in intervalllengths
        ]  # always same-size
        split_intervals = [
            [
                (lower + i * children_lengths[j], lower + (i + 1) * children_lengths[j])
                for i in range(cuts_per_dim)
            ]
            for j, (lower, _) in enumerate(intervals)
        ]
        cut_points = [
            [
                lower + (i * children_lengths[j]) + children_lengths[j] / 2
                for i in range(cuts_per_dim)
            ]
            for j, (lower, _) in enumerate(intervals)
        ]
        cut_points = [list(cut) for cut in itertools.product(*cut_points)]
        split_intervals = [
            list(interval) for interval in itertools.product(*split_intervals)
        ]
        return cut_points, split_intervals


# def euclidean_metric(cutpoints, intervals):
#     """Construct cutpoints and intervallengths for the children based on the
#     Euclidean metric.
#
#     :param [Float] cutpoints: Cut points of parent node for each dimension.
#     :param [(Float, Float)] intervals: Interval bounds of parent node for each dimension.
#     :return: Cutpoints and interval bounds for left and right child.
#     :rtype: [Float], [(Float, Float)], [Float], [(Float, Float)]
#
#     """
#     intervallengths = [u - l for l, u in intervals]
#     maxindices = np.argwhere(intervallengths == np.amax(intervallengths))
#     if len(maxindices) > 1:
#         np.random.seed(28)
#         argmax_dimension = np.random.choice(np.squeeze(maxindices))
#     else:
#         argmax_dimension = maxindices[0][0]
#     new_length = np.asarray(intervallengths)[argmax_dimension] * 0.5
#     left_cut, right_cut = np.asarray(cutpoints, dtype=float), np.asarray(
#         cutpoints, dtype=float
#     )
#     left_i, right_i = intervals.copy(), intervals.copy()
#     left_cut[argmax_dimension] = cutpoints[argmax_dimension] - new_length * 0.5
#     right_cut[argmax_dimension] = cutpoints[argmax_dimension] + new_length * 0.5
#     left_i[argmax_dimension] = (
#         intervals[argmax_dimension][0],
#         cutpoints[argmax_dimension],
#     )
#     right_i[argmax_dimension] = (
#         cutpoints[argmax_dimension],
#         intervals[argmax_dimension][1],
#     )
#     return [list(left_cut), list(right_cut)], [list(left_i), list(right_i)]


def distance(kernelfunction):
    """Kernel induced distance function.

    :param Function kernelfunction: Kernelfunction.
    :return: Distancefunction.
    :rtype: Function.

    """

    def distance_function(x1, x2):
        return np.sqrt(
            kernelfunction(x1, x1) + kernelfunction(x2, x2) - 2 * kernelfunction(x1, x2)
        )

    return distance_function


def center1d(bounds, params, i):
    """Find the center point along one dimension.

    :param [Float] lowest_corner: Coordinates of one point along each dimension.
    :param Int dimension: Along which dimension to find the center.
    :param (Float, Float) bounds: Upper and lower boundary for that dimension.
    :param Parameters params: Experimental configuration.
    :param Int i: index for kernel function.
    :return: Coordinates of center point, distance to bounds
    :rtype: center, length

    """
    lower, upper = bounds
    move_vector = np.linspace(lower, upper, 100)
    grid = move_vector
    kernelfunction = kernel_functions.get_my_kernel_by_name(params, i)
    distancefunction = distance(kernelfunction)
    distances = []
    for point in grid:
        distance_lower = distancefunction(np.asarray(lower), np.asarray(point))
        distance_upper = distancefunction(np.asarray(upper), np.asarray(point))
        distances.append(np.min([distance_lower, distance_upper]))
    argmax_point = np.argmax(distances)
    return grid[argmax_point], np.max(distances)


def distance_matrix(kernel, grid, nearby_coordinates):
    """Calculate distances between points on grid and nearby_points.
    Only for stationary kernels.

    :param probnum.kernels kernel: kernelfunction.
    :param [(Float, Float)] intervals: Upper and lower bounds of interval for all dimensions.
    :param [([Float], Float)] nearby_points: Coordinates and function value for nearby observed
                                                                                     points.
    :return: Distances between nearby points and points on grid.
    :rtype: np.array[Float]

    """
    covariances = kernel.matrix(nearby_coordinates, grid)
    # the following only holds for stationary kernels
    variance = kernel.matrix(np.ones(kernel.input_shape), np.ones(kernel.input_shape))
    distances = np.sqrt(2 * variance - 2 * covariances)
    return distances


def get_center(intervals, params, i):
    """Find the center point of a cell.

    :param [(Float, Float)] intervals: Upper and lower boundaries along each dimension.
    :param Parameters params: Experimental configuration.
    :param Int i: Index of kernelfunction.
    :return: Coordinate of center, Coordinate along max dimension, max dimension
    :rtype: [Float], Float, Int

    """
    center, lengths = np.zeros(params.dim), np.zeros(params.dim)
    for dimension in range(0, params.dim):
        center_i, length = center1d(intervals[dimension], params, i)
        center[dimension] = center_i
        lengths[dimension] = length
    argmax_dimension = np.argmax(lengths)
    cutpoint = center[argmax_dimension]
    return center, cutpoint, argmax_dimension


def numerical_metric(intervals, params, i):
    """Construct cutpoints and intervallengths for the children based on
    numerical optimization in one dimension.

    :param [(Float, Float)] intervals: Interval bounds of parent node for each dimension.
        :param Parameters params: Experimental configuration.
        :param Int i: Index of kernelfunction.
        :param [(Float, Float)] intervals: Interval bounds of parent node for each dimension.
    :return: Cutpoints and intervallengths for left and right child.
    :rtype: [Float], [Float], [Float], [Float]

    """
    assert params.nb_children == 2
    _, cutpoint, argmax_dimension = get_center(intervals, params, i)
    left_i, right_i = intervals.copy(), intervals.copy()
    left_i[argmax_dimension] = (intervals[argmax_dimension][0], cutpoint)
    right_i[argmax_dimension] = (cutpoint, intervals[argmax_dimension][1])
    left_cutpoint, _, _ = get_center(left_i, params, i)
    right_cutpoint, _, _ = get_center(right_i, params, i)
    return [left_cutpoint, right_cutpoint], [left_i, right_i]
