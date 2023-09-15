"""Experiment with random search."""
import sys
import random

sys.path.append(".")
import numpy as np
import utils
import time
from experiment import GTExperiment, RealExperiment



np.random.seed(28)


class RandomExperiment(GTExperiment, RealExperiment):
    """Experiment with random selection of function values."""

    def __init__(self, params):
        """Experiment with random selection of function values.

        :param Parameters params: Experimental configuration.
        :return: None
        :rtype: None

        """
        if params.benchmark == "groundtruth":
            GTExperiment.__init__(self, params)
        else:
            RealExperiment.__init__(self, params)

    def run_random_search(self, sample):
        """Run the random optimization for the given sample and log the
        results.

        :param np.array[Float] sample: Sample.
        :return: None.
        :rtype: None

        """
        names = ["randomregret", "randomtimelogs"]
        log_files = utils.open_log_files(self, names)
        if self.params.benchmark == "groundtruth":
            random_regret, time_logs = self.run_random_gt(sample)
        else:
            random_regret, time_logs = self.run_random_real()
        utils.write_logs(log_files["randomregret"], random_regret)
        utils.write_logs(log_files["randomtimelogs"], time_logs)

    def run_random_real(self):
        """Random selection of function evaluations.

        :return: function evaluations, time per function evaluation
        :rtype: [Float], [Float]

        """
        evaluations = []
        time_logs = []
        for _ in range(self.params.steps):
            start = time.time()
            random_x = [
                np.asarray([random.uniform(interval[0], interval[1])])
                for interval in self.params.input_range
            ]
            try:
                random_y = self.benchmark_function[0](random_x)[0][0]
            except:
                random_y = self.benchmark_function[0](random_x)
            end = time.time()
            evaluations.append(random_y)
            time_logs.append(end - start)
        return evaluations, time_logs

    def run_random_gt(self, sample):
        """Randomly select function values.

        :param np.array[float] sample: Sampled function.
        :return: Simple regret for all steps.
        :rtype: [Float]

        """
        regret = []
        time_logs = []
        for _ in range(self.params.steps):
            start = time.time()
            random_y = np.random.choice(sample.ravel())
            end = time.time()
            regret.append(self.regret(random_y, sample))
            time_logs.append(end - start)
        return regret, time_logs

    def run_experiment(self):
        """Run the optimization process for all functions.

        :return: None
        :rtype: None
        """
        if self.params.benchmark == "groundtruth":
            for i, sample in enumerate(self.samples):
                self.run_random_search(sample)
        else:
            self.run_random_search(None)
