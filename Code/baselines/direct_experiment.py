"""Experiment with DIRECT for Bayesian Optimization."""
import sys

sys.path.append(".")
import numpy as np
from baselines.DiRect import DiRect

import utils
from experiment import GTExperiment, RealExperiment
import benchmark_functions
import time
import matplotlib.pyplot as plt



np.random.seed(28)


class DirectExperiment(GTExperiment, RealExperiment):
    """Epxeriment with direct.

    :param Parameters params: Experimental configuration.
    :attr [Floats] direct_observations: Function evaluations during search.

    """

    def __init__(self, params):
        """Epxeriment with direct.

        :param Parameters params: Experimental configuration.
        :return: None.
        :rtype: None

        """
        if params.benchmark == "groundtruth":
            GTExperiment.__init__(self, params)
        else:
            RealExperiment.__init__(self, params)
        self.direct_observations = []

    def get_direct_f(self, sample):
        """Function to optimize in format for direct.

        :param np.array[Float] sample: Function to optimize.
        :return: Function to optimize.
        :rtype: ?

        """
        if self.params.benchmark == "groundtruth":
            lower_bounds = np.asarray([l for l, u in self.params.input_range])
            upper_bounds = np.asarray([u for l, u in self.params.input_range])

            return benchmark_functions.DirectSample(
                lower_bounds, upper_bounds, sample, self.params.input_range, self.params
            )

        def direct_fun(x):
            x = x[None, :]
            if self.params.max_or_min == "min":
                return float(self.benchmark_function[1](x)[0])
            else:
                return -float(self.benchmark_function[1](x)[0])

        return direct_fun

    def direct_results(self, yvalues, sample):
        """Calculate simple regret for the function evaluations.

        :param [Float] yvalues: Function evaluations.
        :param np.array[Float] sample: Functionbeing optimized.
        :return: Simple regret for each step.
        :rtype: [Float]

        """
        regret = [self.regret(yvalues[i], sample) for i in range(len(yvalues))]
        return regret

    def run_direct_search(self, sample):
        """Run TurBO for the given sample.

        :param ? sample: Function to be optimized.
        :return: Time per step, evaluated function values, simple regret, ?
        :rtype: ?, ?, [Float], Int

        """
        fun = self.get_direct_f(sample)
        search_space = np.asarray(
            [[a, b] for (a, b) in self.params.input_range]
        ).reshape(-1, 2)
        direct = DiRect(
            fun, search_space, max_feval=self.params.steps, max_iter=self.params.steps
        )
        direct.run()
        yobs = []
        for hist in direct.l_hist:
            yobs.append(hist[1])
        regret = self.direct_results(yobs, sample)
        mean_it_time = direct.iteration_times
        return mean_it_time, yobs, regret

    def run_direct(self, sample):
        """Run direct for the given sample + logging.

        :param ? sample: Function to be optimized.
        :return: None.
        :rtype: None

        """
        names = [
            "direct" + "_regret",
            "direct" + "_observations",
            "directtimelogs",
            "directevals",
        ]
        log_files = utils.open_log_files(self, names)
        (
            times_per_step,
            direct_observations,
            direct_regret,
        ) = self.run_direct_search(sample)
        self.direct_observations.append(direct_observations)
        utils.write_logs(
            log_files["direct" + "_regret"], direct_regret[: self.params.steps]
        )
        utils.write_logs(
            log_files["direct" + "_observations"],
            direct_observations[: self.params.steps],
        )
        utils.write_timelogs_direct(log_files["directtimelogs"], times_per_step)
        print(
            "direct_regret",
            np.min(direct_regret),
            len(direct_regret),
            np.sum(times_per_step),
        )

    def run_experiment(self):
        """Run direct for all samples.

        :return: None.
        :rtype: None

        """
        for i, sample in enumerate(self.samples):
            print("sample", i)
            self.run_direct(sample)
