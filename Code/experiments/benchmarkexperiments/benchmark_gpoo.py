"""Run GP-OO on the benchmark functions for different choices of beta."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from GPOO.gpoo_experiment import GPOOExperiment
import benchmark_functions
import utils

random.seed(28)

NB_DOMAINS = 10


def specify_experimental_configuration(benchmark, sampled_domain, beta, i):
    """Specify the experimental configuration for a benchmark experiment."""
    steps = 10000
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
        beta=beta,
        steps=steps,
        max_or_min="min",
        mode="greedy",
        ucb_discretization=None,
    )
    domainparams = parameters.Domainparameters(
        input_range=sampled_domain,
        nb_samples=1,
        step_size=1,
        benchmark=benchmark,
        discretization=1,
    )
    exp_id = (
        benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"
    )
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_benchmark_experiments():
    """Run the experiments with GP-UCB."""
    for benchmark, domains in benchmark_functions.SAMPLED_DOMAINS.items():
        print(benchmark)
        for i, domain in enumerate(domains):
            print(benchmark, domain)
            for beta in [100, 10, 1, 0.1]:
                exp_id, params = specify_experimental_configuration(
                    benchmark, domain, beta, i
                )
                gpooexp = GPOOExperiment(params)
                gpooexp.generate_samples()
                gpooexp.run_experiment()


run_benchmark_experiments()
