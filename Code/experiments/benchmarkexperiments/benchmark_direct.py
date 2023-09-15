"""Run Direct on the benchmark functions."""
import os
import random
import numpy as np
import pandas as pd
import time


import parameters
import benchmark_functions
from baselines.direct_experiment import DirectExperiment
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
        jitter=None,
        mode="greedy",
        adabkb_ucb_beta=None,
    )
    domainparams = parameters.Domainparameters(
        input_range=sampled_domain,
        nb_samples=1,
        step_size=1,
        benchmark=benchmark,
        discretization=1,
    )
    exp_id = benchmark + "_domain" + str(i) + "direct" + "_tmlr_rebuttal_with_times2"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_benchmark_experiments():
    """Run the experiments with Direct"""
    for benchmark, domains in benchmark_functions.SAMPLED_DOMAINS.items():
        for i, domain in enumerate(domains):
            print(benchmark, domain)
            exp_id, params = specify_experimental_configuration(benchmark, domain, i)
            directexp = DirectExperiment(params)
            directexp.generate_samples()
            directexp.run_experiment()


run_benchmark_experiments()
