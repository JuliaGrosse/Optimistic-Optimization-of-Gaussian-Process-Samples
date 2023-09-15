"""Run TurBO on the benchmark functions."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from baselines.turbo_experiment import TurBOExperiment
import benchmark_functions
import utils

random.seed(28)

NB_DOMAINS = 10


def specify_experimental_configuration(benchmark, sampled_domain, i):
    """Specify the experimental configuration for a benchmark experiment."""
    lengthscale = benchmark_functions.LENGTHSCALES[benchmark]
    kernelparams = parameters.Kernelparameters(
        kernelname=["matern"],
        variance=[1],
        lengthscale=[lengthscale],
        c=[0],
    )
    optimizerparams = parameters.Optimizerparameters(
        epsilon=0.05,
        init_ucb=10,
        beta=None,
        steps=10000,
        max_or_min="min",
        ucb_discretization=None,
        mode="greedy",
    )
    domainparams = parameters.Domainparameters(
        input_range=sampled_domain,
        nb_samples=1,
        step_size=1,
        benchmark=benchmark,
        discretization=1,
    )
    exp_id = benchmark + "_domain" + str(i) + "_tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_benchmark_experiments():
    """Run the experiments with TurBO."""
    for benchmark, domains in benchmark_functions.SAMPLED_DOMAINS.items():
        print(benchmark)
        for i, domain in enumerate(domains):
            print(benchmark, domain)
            exp_id, params = specify_experimental_configuration(benchmark, domain, i)
            turboexp = TurBOExperiment(params)
            turboexp.generate_samples()
            turboexp.run_experiment()


run_benchmark_experiments()
