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
        adabkb_ucb_beta=None,
        steps=500,
        max_or_min="min",
        ucb_discretization=None,
        jitter=None,
        mode="greedy",
        v1=None,
    )
    domainparams = parameters.Domainparameters(
        input_range=[(0, 1), (0, 1), (0, 1)],
        nb_samples=1,
        step_size=25,
        benchmark="groundtruth",
        discretization=1,
    )
    exp_id = "direct_threedimse_tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_synthetic_experiment():
    """Run the experiments with EI"""
    exp_id, params = specify_experimental_configuration()
    directexp = DirectExperiment(params)
    directexp.load_samples(
        "/Users/juliagrosse/Desktop/FastBO/TMLR/Code/results/groundtruth/['squaredexponential']/samples_threedimse.npy"
    )
    directexp.run_experiment()


print("***********")
run_synthetic_experiment()
