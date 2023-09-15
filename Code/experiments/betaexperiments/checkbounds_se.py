"""Experiment for the choice of beta for 3D samples from GP with squaredexponential."""
import numpy as np
import experiment
import utils
import math
import GPOO.optimizer as optimizer
import GPOO.construct_children_utils as construct_children_utils
import parameters
import GPOO.gpoo_experiment as gpoo_experiment
import experiments.betaexperiments.check_bounds as check_bounds
import matplotlib.pyplot as plt

np.random.seed(1)

print("*** Squared Exponential ***")
### lengthscale 1
KERNEL = parameters.Kernelparameters(
    kernelname=["squaredexponential"], variance=[1], lengthscale=[0.2], c=[None]
)
DOMAIN = parameters.Domainparameters(
    benchmark="groundtruth",
    input_range=[(0, 1), (0, 1), (0, 1)],
    nb_samples=100,
    discretization=1,
    step_size=25,
)
OPTIMIZER = parameters.Optimizerparameters(mode="lengthscale")
PARAMS = parameters.Parameters(
    "betaexperimentonedimselengthscale", KERNEL, DOMAIN, OPTIMIZER
)
LINEAREXP = gpoo_experiment.GPOOExperiment(PARAMS)
LINEAREXP.generate_samples()
print("generate samples")
sample1 = LINEAREXP.samples[0]
print("generated samples")
samples = LINEAREXP.samples.copy()

print(
    "heuristic 1 (lengthscale):",
    check_bounds.percentage_bounds_hold(LINEAREXP, level=10),
)

OPTIMIZER = parameters.Optimizerparameters(mode="heuristic")
PARAMS = parameters.Parameters(
    "betaexperimentonedimseheuristic", KERNEL, DOMAIN, OPTIMIZER
)
LINEAREXP = gpoo_experiment.GPOOExperiment(PARAMS)
LINEAREXP.samples = samples
sample1 = LINEAREXP.samples[0]

print(
    "heuristic 2 (heuristic):", check_bounds.percentage_bounds_hold(LINEAREXP, level=10)
)

OPTIMIZER = parameters.Optimizerparameters(mode="discretization")
PARAMS = parameters.Parameters(
    "betaexperimentonedimsediscretization", KERNEL, DOMAIN, OPTIMIZER
)
LINEAREXP = gpoo_experiment.GPOOExperiment(PARAMS)
LINEAREXP.samples = samples
sample1 = LINEAREXP.samples[0]

print(
    "heuristic 3 (discretization):",
    check_bounds.percentage_bounds_hold(LINEAREXP, level=10),
)
