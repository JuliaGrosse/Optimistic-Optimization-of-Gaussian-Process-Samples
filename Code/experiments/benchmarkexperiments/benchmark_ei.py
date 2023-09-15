"""Run EI on the benchmark functions for different choices of jitter."""
import os
import random
import numpy as np
import pandas as pd

import parameters
from baselines.ei_experiment import EiExperiment
import benchmark_functions
import utils

random.seed(28)

NB_DOMAINS = 10


def specify_experimental_configuration(benchmark, sampled_domain, jitter, i):
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
        steps=200,
        max_or_min="min",
        ucb_discretization=None,
        jitter=jitter,
        mode="greedy",
    )
    domainparams = parameters.Domainparameters(
        input_range=sampled_domain,
        nb_samples=1,
        step_size=1,
        benchmark=benchmark,
        discretization=1,
    )
    exp_id = benchmark + "_domain" + str(i) + "_jitter" + str(jitter) + "_tmlr"
    params = parameters.Parameters(exp_id, kernelparams, domainparams, optimizerparams)
    return exp_id, params


def run_benchmark_experiments():
    """Run the experiments with EI"""
    for benchmark, domains in benchmark_functions.SAMPLED_DOMAINS.items():
        print(benchmark)
        for i, domain in enumerate(domains):
            print(benchmark, domain)
            for jitter in [0.01, 0.1, 0.001, 0.0001]:
                exp_id, params = specify_experimental_configuration(
                    benchmark, domain, jitter, i
                )
                eiexp = EiExperiment(params)
                eiexp.generate_samples()
                eiexp.run_experiment()

    # post process logging info
    with open("./results/" + "/logging") as logging_file:
        f_out = None
        for line in logging_file:
            if "Start logging" in line:
                print("start logging")
                get_title = line.split("#")
                benchmark, _ = get_title[-2].split("_domain")
                title = utils.get_logging_filename_benchmarks(
                    benchmark, get_title[-3], get_title[-1], get_title[-2]
                )
                if f_out:
                    f_out.close()
                f_out = open(f"{title}.txt", "w")
            if f_out:
                f_out.write(line)
    if f_out:
        f_out.close()
    os.remove("./results/logging")


run_benchmark_experiments()
