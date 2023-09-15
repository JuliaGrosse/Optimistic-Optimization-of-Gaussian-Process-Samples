"""Utilities to check if the upper bounds hold."""
import numpy as np
import math
from GPOO.optimizer import TreeSearchOptimizer
import GPOO.construct_children_utils as construct_children_utils
import scipy.stats
import utils
import matplotlib.pyplot as plt


####################################################################################################
# ATTENTION: For these experiments I used a smaller noise term for the sampling of the functions:
# 1e-10 instead of 1e-5.
####################################################################################################
def bound_holds(exprmt, center, interval, sample):
    """Check if an intervals' upper bound holds.

    :param GPOOExperiment exprmt: GPOO Experiment.
    :param [Float] center: Coordinate of the center.
    :param [(Float, Float)] interval: Bounds of the interval.
    :param np.array[Float] sample: Sampled function from GP.
    :return: True if bound holds.
    :rtype: Boolean

    """
    optimizer = TreeSearchOptimizer(sample, exprmt.params)
    sample_cell = sample
    for dim in range(exprmt.params.dim):
        left_index = exprmt.calc_cut_point(interval[dim][0], dim=dim)
        right_index = exprmt.calc_cut_point(interval[dim][1], dim=dim)
        sample_cell = sample_cell.take(indices=range(left_index, right_index), axis=dim)
    center_value = optimizer.observation(center)
    max_function_value = np.max(sample_cell)
    diameter = optimizer.diameter_term(interval, center)
    beta = optimizer.calculate_beta(interval, None)
    return center_value + beta * diameter >= max_function_value


def all_bounds_hold(exprmt, center, interval, sample, level):
    """Check if the bounds holds for intervals up to a certain depth of the tree.

    :param GPOOExperiment exprmt: GPOO Experiment.
    :param [Float] center: Coordinate of the center from root of (sub)tree.
    :param [(Float, Float)] interval: Interval for the root of (sub)tree.
    :param np.array[Float] sample: Sampled function from GP.
    :param Int level: Depth of the tree.
    :return: True if all bounds including the ones in subtrees hold.
    :rtype: Boolean

    """
    if level == 0:
        return True
    if not bound_holds(exprmt, center, interval, sample):
        return False
    optimizer = TreeSearchOptimizer(exprmt.samples[0], exprmt.params)
    children_centers, intervals = optimizer.construct_children(center, interval)
    left_bounds_hold = all_bounds_hold(
        exprmt, children_centers[0], intervals[0], sample, level - 1
    )
    right_bounds_hold = all_bounds_hold(
        exprmt, children_centers[1], intervals[1], sample, level - 1
    )
    return left_bounds_hold and right_bounds_hold


def percentage_bounds_hold(exprmt, level=10):
    """Percentage of not-broken bounds (not per sample, for all samples).

    :param GPOOExperiment exprmt: GPOO Experiment.
    :param Int level: Depth of the tree. Defaults to 10.
    :return: Percentage of broken bounds.
    :rtype: Float

    """
    not_broken = 0
    center, _ = construct_children_utils.construct_root_node(exprmt.params)
    for i in range(exprmt.params.nb_samples):
        not_broken += all_bounds_hold(
            exprmt, center, exprmt.params.input_range, exprmt.samples[i], level
        )
    return (not_broken / exprmt.params.nb_samples) * 100
