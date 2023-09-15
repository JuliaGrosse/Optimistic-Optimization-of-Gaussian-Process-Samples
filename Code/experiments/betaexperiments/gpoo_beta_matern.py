"""Run GPOO on the GP samples."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from GPOO.gpoo_experiment import GPOOExperiment
import benchmark_functions
import utils

random.seed(123)

####################################################################################################
# Attention:
# Make sure the path to load the sampled functions is specified corretly.
# Alternatively you can generate a new set of samples, by replacing the line
# ucbexp.load_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
# with
# ucbexp.generate_samples(".../results/groundtruth/['squaredexponential']/samples_threedimse.npy")
####################################################################################################


def specify_experimental_configuration(mode, beta):
    """Specify the experimental configuration"""
    kernelparams = parameters.Kernelparameters(
        kernelname=["matern"],
        variance=[1],
        lengthscale=[0.2],
        c=[0],
    )
    optimizerparams = parameters.Optimizerparameters(
        epsilon=0.05,
        init_ucb=10,
        beta=beta,
        steps=1000,
        max_or_min="min",
        ucb_discretization=None,
        jitter=None,
        mode=mode,
    )
    domainparams = parameters.Domainparameters(
        input_range=[(0, 1), (0, 1), (0, 1)],
        nb_samples=20,
        step_size=25,
        benchmark="groundtruth",
        discretization=1,
    )
    exp_id = "gpoo_betaexperiment_" + mode + "_beta" + str(beta) + "tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_synthetic_experiment():
    """Run the experiments with TurBO."""
    for mode in ["heuristic", "discretization", "lengthscale"]:
        gpoo_id, params = specify_experimental_configuration(mode=mode, beta=None)
        gpooexp = GPOOExperiment(params)
        gpooexp.load_samples(".../groundtruth/['matern']/samples_threedimmatern.npy")
        gpooexp.run_experiment()
    for beta in [0.1, 1, 10, 100]:
        gpoo_id, params = specify_experimental_configuration(mode="greedy", beta=beta)
        gpooexp = GPOOExperiment(params)
        gpooexp.load_samples(
            ".../results/groundtruth/['matern']/samples_threedimmatern.npy"
        )
        gpooexp.run_experiment()


run_synthetic_experiment()
