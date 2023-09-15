"""Optimize a GP sample using tree search."""
import time
import math
import numpy as np
import networkx as nx
import GPOO.maxheap as maxheap
import GPOO.gpoo_diameters as gpoo_diameters
import GPOO.construct_children_utils as construct_children_utils
import utils
import benchmark_functions
import kernel_functions


class TreeSearchOptimizer:
    """Short summary.

    :param np.array[Float] sample: Samples.
    :param Parameters params: Experimental configuration.
    :attr function benchmarkfunction: Function to be optimized.
    :attr np.array[Float] stored_observations: Observed function values.
    :attr np.array[?] stored_nodes: TODO
    :attr np.array[Float] simple_regret: Observed simple regret.
    :attr ? diameter_term: Diameter terms.
    :attr params:
    :attr sample:

    """

    def __init__(self, sample, params):
        """Optimize a sample from a GP with tree search.

        :param [Float] sample: Sample f from a GP.
        :param Parameters params: Experimental configuration.
        :return: None
        :rtype: None

        """

        self.params = params
        if params.benchmark == "groundtruth":
            self.sample = np.squeeze(sample)
            self.benchmarkfunction = None
        else:
            self.sample = None
            self.benchmarkfunction = benchmark_functions.benchmark_function(
                params.benchmark
            )[0]

        self.stored_observations = []
        self.stored_nodes = []
        self.simple_regret = []
        self.diameter_term = gpoo_diameters.get_gpoo_diameter_term(self.params)

    def calculate_beta(self, ilengths, level):
        """Calculate value of the exploration constant 'beta'.

        :param [(Float, Float)] ilengths: Upper and lower boundaries along the dimensions.
        :param Int level: Level of the tree.
        :return: Float.
        :rtype: Float

        """

        if self.params.mode == "greedy":
            return self.params.beta

        def factor(ilength):
            discretization = self.params.discretization
            return ilength * discretization

        if self.params.mode == "discretization":
            number_of_points = np.max(
                [
                    1,
                    self.params.steps
                    * np.prod(
                        [factor(ilength[1] - ilength[0]) for ilength in ilengths]
                    ),
                ]
            )
            return np.sqrt(2 * np.log(2 * number_of_points / (self.params.epsilon)))
        if self.params.mode == "heuristic":
            if self.params.kernelname[0] == "matern":
                constant = 3 / 2
            else:
                constant = 1
            number_of_points = np.max(
                [
                    0,
                    (constant / (self.params.lengthscale[0]))
                    ** self.params.dim  # self.params.steps
                    * np.prod(
                        [factor(ilength[1] - ilength[0]) for ilength in ilengths]
                    ),
                ]
            )
            return np.sqrt(
                2
                * np.log(
                    2
                    * (constant / (self.params.lengthscale[0])) ** self.params.dim
                    * number_of_points
                    / (self.params.epsilon)
                )
            )

        if self.params.mode == "lengthscale":
            number_of_points = np.max(
                [
                    1,
                    (1 / (self.params.lengthscale[0]))
                    ** self.params.dim  # self.params.steps
                    * np.prod(
                        [factor(ilength[1] - ilength[0]) for ilength in ilengths]
                    ),
                ]
            )
            return np.sqrt(
                2
                * np.log(
                    2
                    # * (1/ (self.params.lengthscale[0])) ** self.params.dim
                    * number_of_points
                    / (self.params.epsilon)
                )
            )

    def regret(self, obs):
        """Calculate simple regret.

        :param Float obs: Observation.
        :param np.array[Float] sample: Function.
        :return: Simple regret.
        :rtype: Float

        """
        if self.params.benchmark == "groundtruth":
            if self.params.max_or_min == "max":
                return np.max(self.sample) - obs
            return obs - np.min(self.sample)
        return obs

    def observation(self, cut_points):
        """Observe f(cut_point).

        :param [Float] cut_points: cut points for each dimensions
        :return: f(cut_point).
        :rtype: Float

        """

        def calc_cut_point(cut_point, dim):
            total_distance = (
                self.params.input_range[dim][1] - self.params.input_range[dim][0]
            )
            cut_distance = cut_point - self.params.input_range[dim][0]
            distance = cut_distance / total_distance
            return np.min(
                [int(distance * self.sample.shape[dim]), self.sample.shape[dim] - 1]
            )

        if self.params.benchmark == "groundtruth":
            indices = np.array(
                [
                    calc_cut_point(cut_point, dim)
                    for dim, cut_point in enumerate(cut_points)
                ]
            )
            obs = self.sample[tuple(indices)]
        else:
            obs = np.squeeze(
                self.benchmarkfunction(np.expand_dims(np.asarray(cut_points), axis=1))
            )
        self.stored_observations.append((cut_points, obs))
        self.simple_regret.append(self.regret(obs))
        return obs

    def utility(self, observation, ilengths, depth, cutpoint):
        """Utility U(x) of a node x.

        :param Float observation: f(x)
        :param [(Float, Float)] ilengths: Upper and lower interval boundary
                                                 for all dimensions
        :return: Utility
        :rtype: Float

        """
        diameter = self.diameter_term(ilengths, cutpoint)
        beta = self.calculate_beta(ilengths, depth)
        exploration = beta * diameter
        if self.params.max_or_min == "max":
            return observation + exploration
        return -1 * (
            observation - exploration
        )  # multiply with -1 since this will be instered into a maxheap

    def evaluate_point(self, obs, cutpoints, intervallengths, depth):
        """Get observation and calculate utility of a point.

        :param [Float] cut_points: cut_points for all dimensions
        :param [(Float, Float)] intervallengths: Upper and lower interval boundary
                                                 for all dimensions
        :return: (U(x), x, f(x), depth(x))
        :rtype: (Float, Float, Float, Int)

        """
        uti = self.utility(obs, intervallengths, depth, cutpoints)
        self.stored_nodes.append((cutpoints, uti, intervallengths))
        return (uti, cutpoints, obs, intervallengths, depth)

    def construct_children(self, cutpoints, intervallengths):
        """Construct cutpoints and intervallengths for the children.

        :param [Float] cutpoints: Cut points of parent node for each dimension.
        :param [Float] intervallengths: Intervallengths of parent node for each dimension.
        :return: Cutpoints and Intervallenghts for left and right child.
        :rtype: [Float], [Float], [Float], [Float]

        """
        if self.params.partition == "black_box":
            return construct_children_utils.numerical_metric(
                intervallengths, self.params, 0
            )
        if self.params.partition == "euclidean":
            return construct_children_utils.euclidean_metric(
                cutpoints, intervallengths, self.params
            )
        print("Partitioning scheme not implemented!")
        raise ValueError()

    def optimize(self, root_node):
        """Optimize the sample f.

        :param ([Float], [Float]): cutpoints and intervallengths for the root node
        :return: Observed function values and simple regret.
        :rtype: [Float],[Float]

        """

        # Initialization
        pointsleft = self.params.steps
        root = self.evaluate_point(0, root_node[0], root_node[1], 0)
        max_heap = maxheap.MaxHeap(self.params.steps, 0)
        max_heap.insert(root)
        time.sleep(0.1)
        storegraph = nx.DiGraph()
        storegraph.add_nodes_from([utils.make_nx_node(root, str(root[1]))])
        time_per_steps = []
        evaluations = 1

        while evaluations + self.params.nb_children <= self.params.steps:

            start = time.time()
            bestnode = max_heap.extractMax()
            _, cutpoints, parent_obs, intervallengths, depth = bestnode
            construction_start = time.time()
            cuts, intervals = self.construct_children(cutpoints, intervallengths)

            construction_end = time.time()

            evaluation_start = time.time()
            length_regret = len(self.simple_regret)

            if self.params.nb_children % 2 == 0:
                observations = [self.observation(cut) for cut in cuts]
                evaluations += self.params.nb_children
            else:
                observations = []
                for i in range(self.params.nb_children):
                    if i == np.floor(self.params.nb_children / 2):
                        observations.append(parent_obs)
                    else:
                        observations.append(self.observation(cuts[i]))
                        evaluations += 1

            evaluation_end = time.time()

            construction2_start = time.time()
            children = []
            for child_i in range(self.params.nb_children):
                child = self.evaluate_point(
                    observations[child_i], cuts[child_i], intervals[child_i], depth + 1
                )
                children.append(child)
            construction2_end = time.time()

            for child in children:
                max_heap.insert(child)
                node = utils.make_nx_node(child, str(child[1]))
                storegraph.add_nodes_from([node])
            pointsleft -= self.params.nb_children
            end = time.time()
            time_per_steps.append(
                [
                    end - start,
                    construction_end
                    - construction_start
                    + construction2_end
                    - construction2_start,
                    evaluation_end - evaluation_start,
                ]
            )
        if len(self.simple_regret) > self.params.steps:
            print("simple_regret", len(self.simple_regret))
            raise ValueError()
        return (
            self.stored_observations,
            self.simple_regret,
            self.stored_nodes,
            (max_heap, storegraph),
            time_per_steps,
        )
