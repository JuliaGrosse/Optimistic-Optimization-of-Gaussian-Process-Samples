"""Experiment with GPOO."""
import time
import sys

sys.path.append(".")
import utils
import numpy as np
from GPOO.optimizer import TreeSearchOptimizer
from GPOO import construct_children_utils
from experiment import GTExperiment, RealExperiment


# import warnings
# warnings.simplefilter('error', RuntimeWarning)

np.random.seed(28)


class GPOOExperiment(GTExperiment, RealExperiment):
    """Experiment with GPOO.

    :param Parameters params: Experimental configuration.
    :attr [[Float]] gpoo_observations: Observed function values for each sample.
    :attr [[Float]] gpoo_stored_nodes: Nodes expanded during search dor each sample (for logging).
    :attr [maxheap] gpoo_stored_maxheaps: Maxheaps built during search for each sample (for logging).

    """

    def __init__(self, params):
        """Experiment with GPOO.

        :param Parameters params: Experimental configuration.
        :return: None.
        :rtype: None

        """
        if params.benchmark == "groundtruth":
            GTExperiment.__init__(self, params)
        else:
            RealExperiment.__init__(self, params)
        self.gpoo_observations = []
        self.gpoo_stored_nodes = []
        self.gpoo_stored_maxheaps = []

    def optimize(self, sample, mode):
        """Optimize the sample.

        :param np.array sample: f
        :param Boolean gpoo: Use HOO (true) or tree search (false)?
        :return: Observed function values, simple regret, expanded nodes, heap, timelogs.
        :rtype: [Float],[Float],[?], maxheap, ?

        """
        root_node = construct_children_utils.construct_root_node(self.params)
        optimizer = TreeSearchOptimizer(sample, self.params)
        observations, regret, stored_node, maxheap, timelogs = optimizer.optimize(
            root_node
        )
        return observations, regret, stored_node, maxheap, timelogs

    def run_gpoo_search(self, sample):
        """Run GPOO on the specified sample.

        :param ? sample: Function/Sample that should be optimized.
        :return: None.
        :rtype: None

        """
        start = time.time()
        mode = self.params.mode
        names = [
            "HOOregret" + mode,
            "HOOobservations" + mode,
            "HOOtimelogs" + mode,
        ]
        log_files = utils.open_log_files(self, names)
        (
            gpoo_observations,
            gpoo_simple_regret,
            gpoo_stored_nodes,
            maxheap,
            time_logs,
        ) = self.optimize(sample, mode)
        self.gpoo_observations.append(gpoo_observations)
        self.gpoo_stored_nodes.append(gpoo_stored_nodes)
        self.gpoo_stored_maxheaps.append(maxheap)
        utils.write_logs(log_files["HOOregret" + mode], gpoo_simple_regret)
        utils.write_logs(log_files["HOOobservations" + mode], gpoo_observations)
        utils.write_timelogs(log_files["HOOtimelogs" + mode], time_logs)
        end = time.time()
        print(
            "go oo regret",
            np.min(gpoo_simple_regret),
            len(gpoo_simple_regret),
            end - start,
        )

    def run_experiment(self):
        """Run the optimization process for all functions/samples.

        :param String mode: Mode for beta.
        :return: None.
        :rtype: None

        """
        for i, sample in enumerate(self.samples):
            print("sample", i)
            self.run_gpoo_search(sample)
