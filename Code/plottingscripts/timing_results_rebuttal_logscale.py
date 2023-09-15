"""Figure with timing results on synthetic functions for Squared Exponential and Matern."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import benchmark_functions

from tueplots import axes, bundles

bundles.tmlr2023()


Evaluation_costs = 0


def _subtract_times(stored_times, times1_name, times2_name):
    """Helper function to read the timing results of GP-UCB."""
    times1 = stored_times[times1_name][-1]
    if times2_name == "iterations2":
        times2 = stored_times["iterations"][-1][1:]
    else:
        times2 = stored_times[times2_name][-1]
    timediff = [time2 - time1 for (time1, time2) in zip(times1, times2)]
    return timediff


def _store_times(stored_times, costs=0):
    """Helper function to read the timing results of GP-UCB."""
    iteration_times = _subtract_times(stored_times, "iterations", "iterations2")
    updating_times = _subtract_times(stored_times, "updating", "acquisition")
    acquisition_times = _subtract_times(stored_times, "acquisition", "evaluation")
    evaluation_times = _subtract_times(stored_times, "evaluation", "iterations2")
    iteration_times = [iteration_time + costs for iteration_time in iteration_times]

    stored_times["iterations"][-1] = iteration_times
    stored_times["updating"][-1] = updating_times
    stored_times["acquisition"][-1] = acquisition_times
    stored_times["evaluation"][-1] = evaluation_times


def _new_sample(stored_times):
    """Helper function to read the timing results of GP-UCB."""
    for keyword in stored_times.keys():
        old = stored_times[keyword]
        old.append([])
        stored_times[keyword] = old
    return stored_times


def read_gp_ucb_times(filename, costs=0):
    """Collect the times GP-UCB needed for each iteration, the updating of the GP-Posterior,
    the optimization of the acquisition function and the evaluation of the function
    from the logging file."""
    print("open", filename)
    with open(filename) as timelog_file:

        stored_times = {
            "iterations": [[]],
            "updating": [[]],
            "acquisition": [[]],
            "evaluation": [[]],
        }

        skip = False
        for line in timelog_file:
            words = line.split()
            time = int(words[0]) * 0.001  # milliseconds to seconds
            if "Iteration" in line:
                stored_times["iterations"][-1].append(time)
            if "Updating parameters of the model" in line:
                if not skip:
                    stored_times["updating"][-1].append(time)
                else:
                    skip = False
            if "Starting gradient-based optimization" in line:
                stored_times["acquisition"][-1].append(time)
            if "Evaluating user function" in line:
                stored_times["evaluation"][-1].append(time)
            if "Stopped after" in line:
                stored_times["iterations"][-1].append(time)
                skip = True

                _store_times(stored_times, costs)
                _new_sample(stored_times)
    return stored_times


def read_hoo_times(filename, costs):
    """Collect the times hoo needed for each iteration, the construction part of an iteration
    and the evaluation part of an iteration from the logging file."""
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times, all_construction_times, all_evaluation_times = [], [], []
        for line in timelog_file:
            iteration_times, construction_times, evaluation_times = [], [], []
            words = line.split("#")
            for i, word in enumerate(words):
                if i % 3 == 0:
                    iteration_times.append(float(word) + costs)
                if i % 3 == 1:
                    construction_times.append(float(word))
                if i % 3 == 2:
                    evaluation_times.append(float(word) + costs)
            all_iteration_times.append(iteration_times)
            all_construction_times.append(construction_times)
            all_evaluation_times.append(evaluation_times)
    stored_times["iterations"] = all_iteration_times
    stored_times["construction"] = all_construction_times
    stored_times["evaluation"] = all_evaluation_times
    return stored_times


def read_random_times(filename, costs):
    """Collect the times hoo needed for each iteration, the construction part of an iteration
    and the evaluation part of an iteration from the logging file."""
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            words = line.split("#")
            for i, word in enumerate(words):
                iteration_times.append(float(word) + costs)
            all_iteration_times.append(iteration_times)
    stored_times["iterations"] = all_iteration_times
    return stored_times


def read_turbo_times(filename, costs, batchsize=10, nb_init=25):
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            line = line[2:]
            line = line[:-2]
            words = line.split(",")
            for i, word in enumerate(words):
                iteration_time = float(word) + costs
                iteration_times.append(iteration_time)
            all_iteration_times.append(iteration_times)
    stored_times["iterations"] = all_iteration_times
    return stored_times


def get_turbo_data(filename, evals_filename, kernelname, benchmark):
    regret_dframe = pd.read_csv(filename, sep="#", header=None)
    evals_dframe = pd.read_csv(evals_filename, sep="#", header=None)

    def cut(x):
        return x[1:-1]

    regret_dframe = regret_dframe.applymap(cut)
    regret_dframe = regret_dframe.astype(float)

    obs = regret_dframe.shape[1]

    # split the dataframe in dataframe over batches
    i = 0
    j = 0
    dfs = []
    while i < obs:
        df = regret_dframe.iloc[:, i : evals_dframe.iloc[0, j]]
        dfs.append(df)
        i = evals_dframe.iloc[0, j]
        j += 1
    # take the min of each batch
    dfs = pd.concat([df.min(axis=1) for df in dfs], axis=1)
    print(dfs.shape)
    return dfs


def get_direct_data(filename):
    kernelname = "matern"
    regret_dframe = pd.read_csv(filename, header=None, sep="#", names=range(10000))
    regret_dframe = regret_dframe.astype(float)
    return regret_dframe


def get_data(filename, kernelname, benchmark="groundtruth"):
    pathname = "./results/" + benchmark + "/['" + kernelname + "']" + filename
    if "turbo" in filename:
        return get_turbo_data(pathname, kernelname, benchmark)
    if "direct" in filename:
        return get_direct_data(pathname)
    with open(pathname, "r") as temp_f:
        col_count = [len(l.split("#")) for l in temp_f.readlines()]
        column_names = list(range(0, max(col_count)))
    regret_df = pd.read_csv(pathname, sep="#", header=None, names=column_names)
    return regret_df


def calc_average_min_regret(dataframe, nb_iterations):
    """Calculate the average minimal regret."""
    nb_iterations = np.min([nb_iterations, dataframe.shape[-1]])
    average_min_regret = np.zeros(nb_iterations, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(
            np.minimum.accumulate(dataframe.iloc[i])[:nb_iterations]
        )
        average_min_regret += min_simple_regret
        min_simple_regret_list.append(min_simple_regret)

    average_min_regret *= 1 / (dataframe.shape[0])

    # create list with standard deviation
    stds = []
    for i in range(nb_iterations):
        std = np.std([results[i] for results in min_simple_regret_list])
        stds.append(std)
    return list(average_min_regret), stds, min_simple_regret_list


def read_direct_times(filename, costs):
    """Collect the times hoo needed for each iteration, the construction part of an iteration
    and the evaluation part of an iteration from the logging file."""
    stored_times = {}
    with open(filename) as timelog_file:
        all_iteration_times = []
        for line in timelog_file:
            iteration_times = []
            line = line[1:]
            line = line[:-2]
            words = line.split(",")
            for i, word in enumerate(words):
                iteration_times.append(float(word) + costs)
            all_iteration_times.append(iteration_times)
    stored_times["iterations"] = all_iteration_times
    return stored_times


def get_time_data(filename, costs=0):
    if "ucb" in filename or "ei" in filename:
        stored_times_gp = read_gp_ucb_times(filename, costs)
        gp_iterations = np.mean(stored_times_gp["iterations"][:-1], axis=0)
        times = np.cumsum(gp_iterations[:-10])
    elif "turbo" in filename:
        stored_times_turbo = read_turbo_times(filename, costs)
        turbo_iterations = np.mean(np.asarray(stored_times_turbo["iterations"]), axis=0)
        times = np.cumsum(turbo_iterations)
        print("turbo times", times)
    elif "random" in filename:
        stored_times_random = read_random_times(filename, costs)
        random_iterations = np.mean(stored_times_random["iterations"], axis=0)
        times = np.cumsum(random_iterations)
    elif "direct" in filename:
        stored_times_direct = read_direct_times(filename, costs)
        direct_iterations = np.mean(stored_times_direct["iterations"], axis=0)
        times = np.cumsum(direct_iterations)
    else:
        stored_times_hoo = read_hoo_times(filename, costs)
        hoo_iterations = np.mean(stored_times_hoo["iterations"], axis=0)
        times = np.cumsum(hoo_iterations[0:])
    return times


def plot_min_regret_per_time(
    filename,
    axis,
    nb_iterations,
    label,
    kernelname,
    maxmilliseconds,
    benchmark,
    color=None,
    costs=0,
    hoo_batch_size=2,
):

    min_iterations = 1000000
    all_times, all_min_regret = [], []
    for i in range(10):

        true_min = benchmark_functions.MINIMA[benchmark][1]
        if "turbo" in filename[0][i]:
            data = get_turbo_data(filename[0][i], filename[2][i], kernelname, benchmark)
            min_regret, stds, _ = calc_average_min_regret(data, data.shape[1])
            print("get_time_data", filename[1][i])
            times = get_time_data(filename[1][i], costs)
            # times = times * 1000 # Times should be in ms
        else:
            data = get_data(filename[0][i], kernelname, benchmark)
            min_regret, stds, _ = calc_average_min_regret(data, nb_iterations)
            print("get_time_data", filename[1][i])

            if "random" in filename[0][i]:
                random_costs = costs / hoo_batch_size
                times = get_time_data(filename[1][i], random_costs)[:nb_iterations]
            else:
                times = get_time_data(filename[1][i], costs)

            if "HOO" in filename[0][i]:
                times = np.repeat(times, hoo_batch_size)

            # times = times * 1000 # Times should be in ms
        min_simple_regret = -(true_min - np.asarray(min_regret))
        min_simple_regret[min_simple_regret <= 1 / 2 ** 20] = 1 / 2 ** 20
        scaled_min_simple_regret = np.log(min_simple_regret)

        all_times.append(np.log(times))
        all_min_regret.append(scaled_min_simple_regret)
        min_iterations = np.min([min_iterations, times.shape[0]])

        if color:
            axis.plot(
                np.log(times),
                scaled_min_simple_regret[: len(times)],
                label=label,
                linewidth=1,
                alpha=0.2,
                color=color,
            )
        else:
            axis.plot(
                np.log(times),
                np.asarray(np.log(min_regret)),
                label=label,
                linewidth=0.3,
            )

    all_times = [times[:min_iterations] for times in all_times]
    all_min_regret = [min_regret[:min_iterations] for min_regret in all_min_regret]
    axis.plot(
        np.mean(np.asarray(all_times), axis=0),
        np.mean(np.asarray(all_min_regret), axis=0),
        label=label,
        linewidth=2,
        alpha=1,
        color=color,
    )
    # add error bars for the standard deviation
    # axis.set_yscale("symlog", linthreshy=0.001)  # or symlog for some plots
    # axis.set_yscale("log")
    # axis.set_xscale("log")  # or symlog/log or nothing for some plots
    # plt.axhline(0, color="black")
