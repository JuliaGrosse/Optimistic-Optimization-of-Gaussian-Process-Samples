"""Results of the hyperparameter grid searches."""
import benchmark_functions
import pandas as pd
import numpy as np

import json

NB_DOMAINS = 10

GPOO_BETAS = {}
GPUCB_BETAS = {}
EI_JITTER = {}


def get_benchmark_filenames_ei(benchmark, i, jitter):
    return benchmark + "_domain" + str(i) + "_jitter" + str(jitter) + "_aistats"


def get_benchmark_filenames_ucb(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_aistats"


def get_benchmark_filenames_turbo(benchmark, i):
    return benchmark + "_domain" + str(i) + "_aistats_small_batch_verylong"



def get_benchmark_filenames_gpoo(benchmark, i, beta):
    return (
        benchmark
        + "_domain"
        + str(i)
        + "_beta"
        + str(beta)
        + "_tmlr_rebuttal_good_implementation"
    )


def get_benchmark_filenames_gpoo_more_steps(benchmark, i, beta):
    return benchmark + "_domain" + str(i) + "_beta" + str(beta) + "_tmlr_rebuttal_good_implementation"


def get_turbo_data(benchmark, experimentname, filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        sep="#",
        header=None,
    )

    def cut(x):
        return x[1:-1]

    regret_dframe = regret_dframe.applymap(cut)
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_data(benchmark, experimentname, filename):
    """Load the results for the given kernel and lengthscale."""
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        sep="#",
        header=None,
    )
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_ada_bkb_data(benchmark, experimentname, filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(
        "./results/"
        + benchmark
        + "/"
        + "['"
        + kernelname
        + "']"
        + "/"
        + filename
        + experimentname
        + ".txt",
        header=None,
        sep="#",
        names=range(10000),
    )
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def best_hyperparameters():
    """Plot the minimal regret per step for GP-OO and GP-UCB on the benchmark functions."""
    benchmarks_and_domains = list(benchmark_functions.ADAPTED_DOMAINS.items())[:12]
    for benchmark, domain in benchmarks_and_domains:
        steps = 200
        true_min = benchmark_functions.MINIMA[benchmark][1]

        ### Find best parameter for GP-UCB
        average_best_min_regrets = []
        betas = [0.1, 1, 10, 100]
        for beta in betas:
            average_best_min_regret = 0
            for i in range(NB_DOMAINS):
                id_gpucb = get_benchmark_filenames_ucb(benchmark, i, beta=beta)
                dataframe = get_data(benchmark, id_gpucb, "ucb_regret")

                # calculate the average of the minimal regret
                average_min_regret = np.zeros(steps, dtype=float)
                min_simple_regret_list = []
                for i in range(dataframe.shape[0]):
                    min_function_values = np.squeeze(
                        np.minimum.accumulate(dataframe.iloc[i])[:steps]
                    )

                    min_simple_regret = -(true_min - min_function_values)
                    min_simple_regret[min_simple_regret <= 0] = 1 / 2 ** 20
                    scaled_min_simple_regret = np.log(min_simple_regret)
                    scaled_positive_min_simple_regret = np.log(min_simple_regret)
                    average_min_regret += scaled_positive_min_simple_regret

                min_simple_regret_list.append(min_simple_regret)

                average_min_regret *= 1 / (dataframe.shape[0])
                average_best_min_regret += average_min_regret.iloc[-1]
            average_best_min_regret /= NB_DOMAINS
            average_best_min_regrets.append(average_best_min_regret)
        argmin_index = np.argmin(np.asarray(average_best_min_regrets))
        best_beta = betas[argmin_index]
        GPUCB_BETAS[benchmark] = best_beta
        print("ucb", best_beta)

        ### Find best hyperparameter for GP-OO
        average_best_min_regrets = []
        betas = [0.1, 1, 10, 100]
        for beta in betas:
            average_best_min_regret = 0
            for i in range(NB_DOMAINS):

                if len(domain) < 4:
                    id_gpoo = get_benchmark_filenames_gpoo(benchmark, i, beta=beta)
                else:
                    id_gpoo = get_benchmark_filenames_gpoo_more_steps(
                        benchmark, i, beta=beta
                    )

                dataframe = get_data(benchmark, id_gpoo, "HOOregretgreedy")
                # calculate the average of the minimal regret
                average_min_regret = np.zeros(dataframe.shape[1], dtype=float)
                min_simple_regret_list = []
                for i in range(dataframe.shape[0]):

                    min_function_values = np.squeeze(
                        np.minimum.accumulate(dataframe.iloc[i])
                    )

                    min_simple_regret = -(true_min - min_function_values)
                    min_simple_regret[min_simple_regret <= 0] = 1 / 2 ** 20
                    scaled_min_simple_regret = np.log(min_simple_regret)
                    scaled_positive_min_simple_regret = np.log(min_simple_regret)
                    average_min_regret += scaled_positive_min_simple_regret

                    min_simple_regret_list.append(min_simple_regret)

                average_min_regret *= 1 / (dataframe.shape[0])
                average_best_min_regret += average_min_regret.iloc[-1]
            average_best_min_regret /= NB_DOMAINS
            average_best_min_regrets.append(average_best_min_regret)
        argmin_index = np.argmin(np.asarray(average_best_min_regrets))
        best_beta = betas[argmin_index]
        GPOO_BETAS[benchmark] = best_beta
        print("gpoo", best_beta, benchmark)
        #
        ### Find best hyperparameter for EI
        steps = 200
        average_best_min_regrets = []
        jitters = [1, 0.1, 0.001, 0.0001]
        for jitter in jitters:
            average_best_min_regret = 0
            for i in range(NB_DOMAINS):
                id_ei = get_benchmark_filenames_ei(benchmark, i, jitter=jitter)
                dataframe = get_data(benchmark, id_ei, "ei_regret")

                # calculate the average of the minimal regret
                average_min_regret = np.zeros(steps, dtype=float)
                min_simple_regret_list = []
                for i in range(dataframe.shape[0]):
                    min_function_values = np.squeeze(
                        np.minimum.accumulate(dataframe.iloc[i])[:steps]
                    )

                    min_simple_regret = -(true_min - min_function_values)
                    min_simple_regret[min_simple_regret <= 0] = 1 / 2 ** 20
                    scaled_min_simple_regret = np.log(min_simple_regret)
                    scaled_positive_min_simple_regret = np.log(min_simple_regret)
                    average_min_regret += scaled_positive_min_simple_regret

                    min_simple_regret_list.append(min_simple_regret)

                average_min_regret *= 1 / (dataframe.shape[0])
                average_best_min_regret += average_min_regret.iloc[-1]
            average_best_min_regret /= NB_DOMAINS
            average_best_min_regrets.append(average_best_min_regret)
        argmin_index = np.argmin(np.asarray(average_best_min_regrets))
        best_jitter = jitters[argmin_index]
        EI_JITTER[benchmark] = best_jitter
        print("ei", best_jitter)




best_hyperparameters()

### Save the best hyperparameter per menthod per domain in a dictionary
with open(
    "./experiments/benchmarkexperiments/benchmarks/gpoo_beta.json", "w"
) as fp:
    json.dump(GPOO_BETAS, fp)

with open("./experiments/benchmarkexperiments/benchmarks/gpucb_beta.json", "w") as fp:
    json.dump(GPUCB_BETAS, fp)

with open("./experiments/benchmarkexperiments/benchmarks/ei_jitter.json", "w") as fp:
    json.dump(EI_JITTER, fp)
