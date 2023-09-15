"""Run AdaBbk on the Gp samples for different choices of beta."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from baselines.direct_experiment import DirectExperiment
import benchmark_functions
import utils


####################################################################################################
# Attention:
# Make sure the path to load the sampled functions is specified corretly.
# Alternatively you can generate a new set of samples, by replacing the line
# eiexp.load_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
# with
# eiexp.generate_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
####################################################################################################


def specify_experimental_configuration(dimension, lengthscale, step_size, steps):
    """Specify the experimental configuration"""
    kernelparams = parameters.Kernelparameters(
        kernelname=["squaredexponential"],
        variance=[1],
        lengthscale=[lengthscale],
        c=[0],
    )
    optimizerparams = parameters.Optimizerparameters(
        epsilon=0.05,
        init_ucb=10,
        beta=None,
        adabkb_ucb_beta=None,
        steps=steps,
        max_or_min="min",
        ucb_discretization=None,
        jitter=None,
        mode="greedy",
        v1=None,
    )
    domainparams = parameters.Domainparameters(
        input_range=[(0, 1)] * dimension,
        nb_samples=20,
        step_size=step_size,
        benchmark="groundtruth",
        discretization=1,
    )
    exp_id = "comparison_dim" + str(dimension) + "_lengthscale_" + str(lengthscale)
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_synthetic_experiment():
    """Run the experiments with EI"""
    for configuration in [
        (3, 1, 40, 1000),
        (3, 0.1, 40, 10000),
        (3, 0.05, 40, 100000),
        (2, 0.5, 100, 1000),
        (2, 0.05, 200, 10000),
        (2, 0.005, 500, 100000),
        (1, 0.5, 1000, 100),
        (1, 0.05, 1000, 10000),
        (1, 0.005, 2000, 100000),
    ]:
        dimension, lengthscale, step_size, steps = configuration
        exp_id, params = specify_experimental_configuration(
            dimension, lengthscale, step_size, steps
        )
        directexp = DirectExperiment(params)
        # directexp.generate_samples()
        directexp.load_samples(
            "/Users/juliagrosse/Desktop/FastBO/TMLR/Code/results/groundtruth/['squaredexponential']/samples_comparison_dim"
            + str(dimension)
            + "_lengthscale_"
            + str(lengthscale)
            + ".npy"
        )
        directexp.run_experiment()


print("***********")
run_synthetic_experiment()
