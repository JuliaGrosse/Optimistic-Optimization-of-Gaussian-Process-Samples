"""Experiment with Expected improvement."""
import logging
import sys

sys.path.append(".")
from GPy.models import GPRegression
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign

import numpy as np

from baselines import expectedimprovement
import utils
from experiment import GTExperiment, RealExperiment

np.random.seed(28)


class EiExperiment(GTExperiment, RealExperiment):
    """Experiment with EI.

    :param Parameters params: Experimental configuration.
    :attr [[Float]] EI_observations: Observed function values for each sample.

    """

    def __init__(self, params):
        """Experiment with EI.

        :param Parameters params: Experimental configuration.
        :return: None
        :rtype: None

        """
        if params.benchmark == "groundtruth":
            GTExperiment.__init__(self, params)
        else:
            RealExperiment.__init__(self, params)
        self.EI_observations = []

    def define_parameter_space(self):
        """Define the input domain for BO.

        :return: Input domain for EI.
        :rtype: ParameterSpace

        """
        parameter_space = ParameterSpace(
            [
                ContinuousParameter("x" + str(i), start, stop)
                for i, (start, stop) in enumerate(self.params.input_range)
            ]
        )
        return parameter_space

    def ei_f(self, sample):
        """Function for EI to optimize = negative (sampled) function.

        :param np.array[Float] sample: (sampled) function.
        :return: (negative) sampled function or benchmark function
        :rtype: Function

        """
        if self.params.benchmark == "groundtruth":

            def f(X):
                if self.params.max_or_min == "max":
                    return np.asarray(
                        [[-self.get_function_value(sample, x)] for x in X]
                    )
                return np.asarray([[self.get_function_value(sample, x)] for x in X])

            return f

        if self.params.max_or_min == "max":
            return -self.benchmark_function[1]
        return self.benchmark_function[1]

    def ei_results(self, bayesopt_loop, sample):
        """Extract results from EI run.

        :param ? bayesopt_loop: emukit object
        :return: observations, simple regret
        :rtype: np.[Floar](steps, 2) np.[Float]

        """
        xpoints = bayesopt_loop.loop_state.X
        yvalues = bayesopt_loop.loop_state.Y
        if self.params.max_or_min == "max":
            yvalues = -yvalues
        regret = [self.regret(yvalues[i][0], sample) for i in range(len(yvalues))]
        return np.concatenate((xpoints, yvalues), axis=1), regret

    def run_ei_search(self, sample):
        """Run EI for a sample f.

        :param np.[Float] sample: sample f
        :return: observations, simple regret
        :rtype: np.[Floar](steps, 2) np.[Float]

        """
        parameter_space = self.define_parameter_space()
        design = RandomDesign(parameter_space)
        random_init = self.params.init_ucb
        X = design.get_samples(random_init)
        f = self.ei_f(sample)
        Y = f(X)
        model_emukit = GPyModelWrapper(
            GPRegression(X, Y, kernel=self.kernel, noise_var=0.0005)
        )

        acquisition_function = expectedimprovement.ExpectedImprovement(
            model=model_emukit, jitter=self.params.jitter
        )

        bayesopt_loop = BayesianOptimizationLoop(
            model=model_emukit,
            space=parameter_space,
            acquisition=acquisition_function,
            batch_size=1,
        )
        bayesopt_loop.run_loop(f, self.params.steps + random_init)
        return self.ei_results(bayesopt_loop, sample)

    def run_ei(self, sample):
        """Run the EI optimization for the given sample and log the
        results.

        :param np.array[Float] sample: Sample.
        :return: None.
        :rtype: None

        """
        names = ["ei_regret", "ei_observations"]
        log_files = utils.open_log_files(self, names)
        ei_observations, ei_regret = self.run_ei_search(sample)
        self.EI_observations.append(ei_observations)
        utils.write_logs(log_files["ei_regret"], ei_regret)
        print("ei_regret", np.min(ei_regret))

    def run_experiment(self):
        """Run the optimization process for all functions.

        :return: None
        :rtype: None
        """

        temp_filename = "./results/" + "/logging"
        logging.basicConfig(
            level=logging.INFO,
            format="%(relativeCreated)6d %(threadName)s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=temp_filename,
            filemode="w",
        )
        logging.info(
            "Start logging #" + str(self.params.kernelname) + "#" + self.params.ID + "#"
        )
        for i, sample in enumerate(self.samples):
            print("sample", i)
            self.run_ei(sample)
