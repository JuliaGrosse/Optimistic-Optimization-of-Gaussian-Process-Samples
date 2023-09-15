"""Container classes for all kinds of parameters."""
import os
import json
import numpy as np


class Kernelparameters:
    """Container class for hyperparameters of a kernel function."""

    def __init__(self, kernelname, variance=1, lengthscale=1, c=1):
        """Container class for hyperparameters of a kernel function.

        If there is more than one kernel specified, their sum will be used.

        :param [String] kernelname: names of used kernels
        :param [Float] variance: variances for kernel functions
        :param [Float] lengthscale: lengthscales for kernel functions
        :param [Float] c: bias terms for kernel functions
        :return: None
        :rtype: None

        """
        self.variance = variance
        self.lengthscale = lengthscale
        self.c = c
        self.kernelname = kernelname


class Domainparameters:
    """Container class for parameters of input domain."""

    def __init__(
        self,
        input_range,
        step_size=1000,
        nb_samples=50,
        benchmark="groundtruth",
        discretization=None,
    ):
        """Container class for parameters of input domain.


        :param [(Float, Float)] input_range: List with ranges for each dimension.
        :param Int step_size: Discretization of input space (for sampling form GP if the
                              benchmark is "groundtruth").
        :param Int nb_samples: Number of sampeled functions f.
        :param String benchmark: Name of the benchmark function.
        :param Int discretization: Discretization along one dimension for calculation of beta.
        :return: None
        :rtype: None

        """
        self.input_range = input_range
        self.step_size = step_size
        self.nb_samples = nb_samples
        self.dim = len(input_range)
        self.benchmark = benchmark
        self.discretization = discretization


class Optimizerparameters:
    """Container class for hyperparameters of the optimizers"""

    def __init__(
        self,
        mode,
        steps=50,
        epsilon=0.05,
        init_ucb=10,
        beta=None,
        adabkb_ucb_beta=None,
        jitter=None,
        max_or_min="min",
        partition="euclidean",
        a=None,
        b=None,
        ucb_discretization=None,
        ucb_has_gradients=True,
        nb_children=2,
        v1=1,
        adabkb_rho=1,
        adabkb_hmax=10,
    ):
        """Container class for hyperparameters of experiment.

        :param Int steps: Number of iterations for the optimizer.
        :param Float epsilon: Works with probability 1 - epsilon.
        :param Int init_ucb: Number of random initialization points for GP UCB.
        :param Float beta: Exploration constant for GP UCB/ GP OO. If None, than it is dervied via
                           the union bound.
        :param Float adabkb_ucb_beta: Exploration constant for GP UCB bound in AdaBkb.
        :param String max_or_min: "max"=maximize the function, "min"=minimize the function
        :param String partition: Euclidean Paritions or Black Box Partitions?
        :param Int a: Constant for GP UCB exploration constant
        :param Int b: Constant for GP UCB exploration constant
        :param Int ucb_discretization: Constant for GP UCB exploration constant
        :param Bool ucb_has_gradients: Does the kernel have gradients? (for UCB acquisition)
        :param Int nb_children: Number for children in GP-OO tree
        :param Float nb_children: >=1, parameter for AdaBkb
        :return: None
        :rtype: None

        """
        self.mode = mode
        self.steps = steps
        self.epsilon = epsilon
        self.init_ucb = init_ucb
        self.beta = beta
        self.adabkb_ucb_beta = adabkb_ucb_beta # TODO: remove this. Not relevant for anything.
        self.max_or_min = max_or_min
        self.partition = partition
        self.a = a
        self.b = b
        self.ucb_discretization = ucb_discretization
        self.ucb_has_gradients = ucb_has_gradients
        self.jitter = jitter
        self.nb_children = nb_children
        self.v1 = v1
        self.adabkb_rho = adabkb_rho # TODO: remove this. Not relevant for anything.
        self.adabkb_hmax = adabkb_hmax # TODO: remove this. Not relevant for anything.


class Parameters:
    """Container class for hyperparameters of experiment."""

    def __init__(
        self,
        ID,
        kernelparameters,
        domainparameters,
        optimizerparameters,
    ):
        """Container class for hyperparameters of experiment.

        :param String ID: Description of parameter `ID`.
        :param Kernelparameters kernelparameters: parameters for kernel.
        :param Domainparameters domainparameters: parameters for domain.
        :param Optimizerparameters optimizerparameters: parameters for optimzer.
        :return: None
        :rtype: None

        """
        self.ID = ID
        self.step_size = domainparameters.step_size
        self.steps = optimizerparameters.steps
        self.epsilon = optimizerparameters.epsilon
        self.nb_samples = domainparameters.nb_samples
        self.input_range = domainparameters.input_range
        self.range_length = self.input_range[0][1] - self.input_range[0][0]
        self.variance = kernelparameters.variance
        self.lengthscale = kernelparameters.lengthscale
        self.c = kernelparameters.c
        self.kernelname = kernelparameters.kernelname
        self.dim = domainparameters.dim
        self.init_ucb = optimizerparameters.init_ucb
        self.beta = optimizerparameters.beta
        self.adabkb_ucb_beta = optimizerparameters.adabkb_ucb_beta
        self.max_or_min = optimizerparameters.max_or_min
        self.partition = optimizerparameters.partition
        self.benchmark = domainparameters.benchmark
        self.discretization = domainparameters.discretization
        self.a = optimizerparameters.a
        self.b = optimizerparameters.b
        self.ucb_discretization = optimizerparameters.ucb_discretization
        self.ucb_has_gradients = optimizerparameters.ucb_has_gradients
        self.jitter = optimizerparameters.jitter
        self.nb_children = optimizerparameters.nb_children
        self.mode = optimizerparameters.mode
        self.v1 = optimizerparameters.v1
        self.adabkb_rho = optimizerparameters.adabkb_rho
        self.nb_children = optimizerparameters.nb_children
        self.adabkb_hmax = optimizerparameters.adabkb_hmax

        params_file_name = (
            "./results/"
            + str(self.benchmark)
            + "/"
            + str(self.kernelname)
            + "/hyperparams_"
            + str(self.ID)
        )
        params = json.dumps(vars(self))
        directory = "./results/" + str(self.benchmark) + "/" + str(self.kernelname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(params_file_name, "w") as outfile:
            outfile.write(params)
