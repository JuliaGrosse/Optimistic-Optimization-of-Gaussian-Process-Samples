"""Experiment with TURBO for Bayesian Optimization."""
import sys

sys.path.append(".")
import numpy as np

import utils
from baselines import turbo_m
from experiment import GTExperiment, RealExperiment
import benchmark_functions


np.random.seed(28)


class TurBOExperiment(GTExperiment, RealExperiment):
    """Epxeriment with TurBO.

    :param Parameters params: Experimental configuration.
    :attr [Floats] turbo_observations: Function evaluations during search.

    """

    def __init__(self, params):
        """Epxeriment with TurBO.

        :param Parameters params: Experimental configuration.
        :return: None.
        :rtype: None

        """
        if params.benchmark == "groundtruth":
            GTExperiment.__init__(self, params)
        else:
            RealExperiment.__init__(self, params)
        self.turbo_observations = []

    def get_turbo_f(self, sample):
        """Function to optimize in format for TurBO.

        :param np.array[Float] sample: Function to optimize.
        :return: Function to optimize.
        :rtype: ?

        """
        if self.params.benchmark == "groundtruth":
            lower_bounds = np.asarray([l for l, u in self.params.input_range])
            upper_bounds = np.asarray([u for l, u in self.params.input_range])

            return benchmark_functions.Sample(
                lower_bounds, upper_bounds, sample, self.params.input_range, self.params
            )
        return self.turbo_benchmark_function

    def turbo_results(self, yvalues, sample):
        """Calculate simple regret for the function evaluations.

        :param [Float] yvalues: Function evaluations.
        :param np.array[Float] sample: Functionbeing optimized.
        :return: Simple regret for each step.
        :rtype: [Float]

        """
        regret = [self.regret(yvalues[i], sample) for i in range(len(yvalues))]
        return regret

    def run_turbo_search(self, sample):
        """Run TurBO for the given sample.

        :param ? sample: Function to be optimized.
        :return: Time per step, evaluated function values, simple regret, ?
        :rtype: ?, ?, [Float], Int

        """
        f = self.get_turbo_f(sample)

        turbo_m_object = turbo_m.TurboM(
            f=f,  # Handle to objective function
            lb=f.lb,  # Numpy array specifying lower bounds
            ub=f.ub,  # Numpy array specifying upper bounds
            n_init=5,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=self.params.steps,  # Maximum number of evaluations
            n_trust_regions=2,  # Number of trust regions
            batch_size=2,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
            params=self.params,
        )
        turbo_m_object.optimize()

        if self.params.max_or_min == "max":
            fX = -turbo_m_object.fX
        else:
            fX = turbo_m_object.fX

        return (
            turbo_m_object.time_per_step,
            fX,
            self.turbo_results(fX, sample),
            turbo_m_object.evals_per_step,
        )

    def run_turbo(self, sample):
        """Run TurBO for the given sample + logging.

        :param ? sample: Function to be optimized.
        :return: None.
        :rtype: None

        """
        names = [
            "turbo" + "_regret",
            "turbo" + "_observations",
            "turbotimelogs",
            "turboevals",
        ]
        log_files = utils.open_log_files(self, names)
        (
            times_per_step,
            turbo_observations,
            turbo_regret,
            turbo_evals,
        ) = self.run_turbo_search(sample)
        self.turbo_observations.append(turbo_observations)
        utils.write_logs(
            log_files["turbo" + "_regret"], turbo_regret[: self.params.steps]
        )
        utils.write_logs(log_files["turboevals"], turbo_evals)
        utils.write_timelogs_turbo(log_files["turbotimelogs"], times_per_step)
        print(
            "turbo_regret",
            np.min(turbo_regret),
            len(turbo_regret),
            np.sum(times_per_step),
        )

    def run_experiment(self):
        """Run TurBO for all samples.

        :return: None.
        :rtype: None

        """
        for i, sample in enumerate(self.samples):
            print("sample", i)
            self.run_turbo(sample)
