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


def plot_cumulative_times(stored_times_gp, stored_times_hoo):
    """Plot the time per iteration for the two methods."""
    plt.rcParams.update(
        figsizes.neurips2022(nrows=1, ncols=2, height_to_width_ratio=0.5)
    )

    ## Plot cumulative times for GP-UCB
    gp_acuisition = np.mean(stored_times_gp["acquisition"][:-1], axis=0)
    gp_updating = np.mean(stored_times_gp["updating"][:-1], axis=0)
    gp_evaluation = np.mean(stored_times_gp["evaluation"][:-1], axis=0)
    gp_iterations = np.mean(stored_times_gp["iterations"][:-1], axis=0)
    gp_overhead = gp_iterations - gp_acuisition - gp_updating - gp_evaluation

    gpucb_df = pd.DataFrame(
        {
            "overhead": gp_overhead[1:],
            "evaluation": gp_evaluation[1:],
            "acquisition": gp_acuisition[1:],
            "GP updating": gp_updating[1:],
        }
    )

    ## Plot times for HOO
    hoo_iterations = np.mean(stored_times_hoo["iterations"], axis=0)
    hoo_construction = np.mean(stored_times_hoo["construction"], axis=0)
    hoo_evaluation = np.mean(stored_times_hoo["evaluation"], axis=0)
    hoo_overhead = hoo_iterations - hoo_construction - hoo_evaluation

    hoo_df = pd.DataFrame(
        {
            "overhead": np.repeat(hoo_overhead[1:], 2) / 2,
            "evaluation": np.repeat(hoo_evaluation[1:], 2) / 2,
            "construction": np.repeat(hoo_construction[1:], 2) / 2,
        }
    )

    fig, axs = plt.subplots(1, 3)

    gpucb_df.plot.area(ax=axs[0])
    hoo_df.plot.area(ax=axs[1])
    axs[1].sharey(axs[0])
    hoo_df.plot.area(ax=axs[2])
    # add axis labels
    fig.supxlabel("iteration")
    fig.supylabel("time")
    axs[0].set_title("GP-UCB")
    axs[1].set_title("GP-OO")
    axs[2].set_title("GP-OO")

    plt.savefig("./plots/timing/timing_area.pdf", bbox_inches="tight")
    plt.show()


def final_cumulative_time_plot(gp_ucb_filename, hoo_filename, kernelname):
    """Compare the cumulative time of the specified two runs."""
    stored_times_gp = read_gp_ucb_times(
        "./results/groundtruth/['" + kernelname + "']/" + gp_ucb_filename
    )

    stored_times_hoo = read_hoo_times(
        "./results/groundtruth/['" + kernelname + "']/" + hoo_filename + ".txt"
    )
    plot_cumulative_times(stored_times_gp, stored_times_hoo)


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


def get_data(filename, kernelname, benchmark="groundtruth"):
    pathname = "./results/" + benchmark + "/['" + kernelname + "']/" + filename
    if "turbo" in filename:
        return get_turbo_data(pathname, kernelname, benchmark)
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
        times = np.cumsum(direct_iterations[0:])
    else:
        stored_times_hoo = read_hoo_times(filename, costs)
        hoo_iterations = np.mean(stored_times_hoo["iterations"], axis=0)
        times = np.cumsum(np.repeat(hoo_iterations[0:], 2))
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
            times = times * 1000  # Times should be in ms
        else:
            data = get_data(filename[0][i], kernelname, benchmark)
            min_regret, stds, _ = calc_average_min_regret(data, nb_iterations)
            print("get_time_data", filename[1][i])

            if "random" in filename[0][i]:
                costs = costs / hoo_batch_size
            times = get_time_data(filename[1][i], costs)

            if "HOO" in filename[0][i]:
                times = np.repeat(times, hoo_batch_size)

            times = times * 1000  # Times should be in ms
        min_regret = -(true_min - np.asarray(min_regret))
        all_times.append(times)
        all_min_regret.append(min_regret)
        min_iterations = np.min([min_iterations, times.shape[0]])

        if color:
            axis.plot(
                times,
                np.asarray(min_regret),
                label=label,
                linewidth=0.5,
                alpha=0.2,
                color=color,
            )
            axis.fill_between(
                times,
                np.asarray(min_regret) - np.asarray(stds),
                np.asarray(min_regret) + np.asarray(stds),
                stds,
                alpha=1,
                color=color,
            )
        else:
            axis.plot(times, np.asarray(min_regret), label=label, linewidth=0.3)
            axis.fill_between(
                times,
                np.asarray(min_regret) - np.asarray(stds),
                np.asarray(min_regret) + np.asarray(stds),
                stds,
                alpha=0.5,
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
    axis.set_yscale("log")
    axis.set_xscale("log")  # or symlog/log or nothing for some plots
    # plt.axhline(0, color="black")


def plot_min_regret_per_time_turbo(
    filename,
    axis,
    nb_iterations,
    label,
    kernelname,
    maxmilliseconds,
    benchmark,
    color=None,
    costs=0,
):
    data = get_data(filename[0], kernelname, benchmark)
    min_regret, stds, _ = calc_average_min_regret(data, nb_iterations)
    times = get_time_data(filename[1], costs)
    if maxmilliseconds is not None:
        times_in_ms = times * 1000
        plot_indices = (times_in_ms < maxmilliseconds).nonzero()[0]
        times = times[plot_indices]
        min_regret = np.asarray(min_regret)[plot_indices]
        stds = np.asarray(stds)[plot_indices]
    times = times * 1000
    if color:
        axis.plot(
            times,
            np.asarray(min_regret),
            label=label,
            linewidth=1,
            alpha=0.5,
            color=color,
        )
        axis.fill_between(
            times,
            np.asarray(min_regret) - np.asarray(stds),
            np.asarray(min_regret) + np.asarray(stds),
            stds,
            alpha=0.5,
            color=color,
        )
    else:
        axis.plot(times, np.asarray(min_regret), label=label, linewidth=0.3)
        axis.fill_between(
            times,
            np.asarray(min_regret) - np.asarray(stds),
            np.asarray(min_regret) + np.asarray(stds),
            stds,
            alpha=0.5,
        )
    # add error bars for the standard deviation
    # axis.set_yscale("symlog")  # or symlog for some plots
    axis.set_yscale("log")
    axis.set_xscale("log")  # or symlog/log or nothing for some plots
    # plt.axhline(0, color="black")


def final_regret_per_time_plot(
    filenames1, filenames2, nb_iterations1, nb_iterations2, costs
):
    fig, axs = plt.subplots(1, 2)
    plot_min_regret_per_time(filenames1, axs[0], nb_iterations1, costs)
    plot_min_regret_per_time(filenames2, axs[1], nb_iterations2, costs)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=4,
    )
    fig.supxlabel("log time in milliseconds")
    fig.supylabel("minimal regret")
    plt.savefig("./plots/timing/timing_regret.png", bbox_inches="tight")
    plt.show()


def plot_regret_helper_refactored(dframe, axis, steps, individual):
    """Helper function to plot the regret."""
    dataframe, label = dframe
    print(dataframe, label)
    steps = np.minimum(steps, dataframe.shape[1])

    # calculate the average of the minimal regret
    average_min_regret = np.zeros(steps, dtype=float)
    min_simple_regret_list = []
    for i in range(dataframe.shape[0]):
        min_simple_regret = np.squeeze(np.minimum.accumulate(dataframe.iloc[i][:steps]))
        average_min_regret += min_simple_regret
        min_simple_regret_list.append(min_simple_regret)
        if individual:
            axis.plot(
                range(len(min_simple_regret)),
                min_simple_regret,
                linestyle="-",
                linewidth="0.5",
                alpha=0.5,
            )

    average_min_regret *= 1 / (dataframe.shape[0])

    # create list with standard deviation
    stds = []
    for i in range(steps):
        std = np.std([results[i] for results in min_simple_regret_list])
        stds.append(std)

    # plot the average minimal regret
    axis.plot(
        range(len(average_min_regret)),
        average_min_regret,
        linestyle="-",
        label=label,
        linewidth="0.5",
    )

    # add error bars for the standard deviation
    if not individual:
        axis.fill_between(
            range(len(average_min_regret)),
            average_min_regret - stds,
            average_min_regret + stds,
            stds,
            alpha=0.5,
        )
    axis.set_yscale("log")
    axis.legend()
