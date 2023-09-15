"""Run EI on the Gp samples for different choices of jitter."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from baselines.ei_experiment import EiExperiment
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


def specify_experimental_configuration(jitter):
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
        jitter=jitter,
        mode="greedy",
    )
    domainparams = parameters.Domainparameters(
        input_range=[(0, 1), (0, 1), (0, 1)],
        nb_samples=20,
        step_size=25,
        benchmark="groundtruth",
        discretization=1,
    )
    exp_id = "ei_threedimse" + "_jitter" + str(jitter) + "_tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_synthetic_experiment():
    """Run the experiments with EI"""
    for jitter in [0.01, 0.1, 0.001, 0.0001]:
        exp_id, params = specify_experimental_configuration(jitter)
        eiexp = EiExperiment(params)
        eiexp.load_samples(
            ".../results/groundtruth/['squaredexponential']/samples_threedimse.npy"
        )
        eiexp.run_experiment()


run_synthetic_experiment()
