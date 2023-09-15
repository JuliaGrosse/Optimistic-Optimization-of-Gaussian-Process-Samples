"""Run TurBO on the GP samples."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from baselines.turbo_experiment import TurBOExperiment
import benchmark_functions
import utils

####################################################################################################
# Attention:
# Make sure the path to load the sampled functions is specified corretly.
# Alternatively you can generate a new set of samples, by replacing the line
# ucbexp.load_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
# with
# ucbexp.generate_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
####################################################################################################


def specify_experimental_configuration():
    """Specify the experimental configuration"""
    kernelparams = parameters.Kernelparameters(
        kernelname=["squaredexponential"],
        variance=[1],
        lengthscale=[0.2],
        c=[0],
    )
    optimizerparams = parameters.Optimizerparameters(
        epsilon=0.05,
        init_ucb=10,
        beta=None,
        steps=1000,
        max_or_min="min",
        ucb_discretization=None,
        jitter=None,
        mode="greedy",
    )
    domainparams = parameters.Domainparameters(
        input_range=[(0, 1), (0, 1), (0, 1)],
        nb_samples=20,
        step_size=25,
        benchmark="groundtruth",
        discretization=1,
    )
    exp_id = "turbo_threedimse_tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_synthetic_experiment():
    """Run the experiments with TurBO"""
    turbo_id, params = specify_experimental_configuration()
    turboexp = TurBOExperiment(params)
    turboexp.load_samples(
        ".../results/groundtruth/['squaredexponential']/samples_threedimse.npy"
    )
    turboexp.run_experiment()


run_synthetic_experiment()
